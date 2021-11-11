import copy

from models.standard_model import StandardModel


class StandardPolicyModel(StandardModel):
    """
    This model inherits StandardModel class
    """

    def __init__(self, dataset, arch, no_grad=False,load_pretrained=False):
        super(StandardPolicyModel, self).__init__(dataset, arch, no_grad, load_pretrained)
        self.init_state_dict = copy.deepcopy(self.state_dict())
        self.factor = 1.0

        # for policy models, we do whiten in policy.net.forward() instead of policy.forward()
        # since _inv models requires grad w.r.t. input in range [0, 1]
        self.net.whiten_func = self.whiten

    def forward(self, adv_image, image=None, label=None, target=None,
                output_fields=('grad', 'std', 'adv_logit', 'logit')):
        # get distribution mean, (other fields such as adv_logit and logit will also be in output)
        output = self.net(adv_image, image, label, target, output_fields)

        # we have two solutions for scaling:
        # 1. multiply scale factor into mean and keep std unchanged
        # 2. keep mean unchanged and make std divided by scale factor

        # since we only optimize mean (if args.exclude_std is True) and we often use some form of momentum (SGDM/Adam),
        # changing the scale of mean will change the scale of gradient and previous momentum may no longer be suitable
        # so we choose solution 2 for std here: std <-- std / self.factor
        if 'std' in output:
            output['std'] = output['std'] / self.factor

        # only return fields requested, since DistributedDataParallel throw error if unnecessary fields are returned
        return {field_key: output[field_key] for field_key in output_fields if field_key in output}

    def reinit(self):
        self.load_state_dict(self.init_state_dict)
        self.factor = 1.0

    def rescale(self, scale):
        self.factor *= scale
