import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['normalization', 'inv_forward']


def normalization(normalization_type, planes, group_size=32):
    # return normalization layer given normalization_type and
    if normalization_type == 'none':
        return nn.Identity()
    elif normalization_type == 'bn':
        return nn.BatchNorm2d(planes)
    elif normalization_type == 'gn':
        assert planes % group_size == 0
        return nn.GroupNorm(num_groups=planes//group_size, num_channels=planes)
    else:
        raise ValueError('Unknown normalization method: {}'.format(normalization))


@torch.enable_grad()
def inv_forward(adv_image, image, label, target, get_logit, normal_mean, empty_coeff, empty_normal_mean,
                training, calibrate, output_fields):
    assert empty_normal_mean.shape[-1] == normal_mean.shape[-1]
    assert adv_image.shape[-1] == image.shape[-1]
    batch_size = adv_image.shape[0]

    output = dict()
    if 'adv_logit' in output_fields or 'grad' in output_fields:
        # run the classification network first, then calculate cw loss, then calculate grad: d(cw_loss) / d(adv_image)
        if 'grad' in output_fields:
            adv_image.requires_grad = True
        adv_logit = get_logit(adv_image)
        if 'adv_logit' in output_fields:
            output['adv_logit'] = adv_logit

        if 'grad' in output_fields:
            # now calculate cw loss: logit_y - logit_t
            # the gradient direction which minimize the loss should point towards the adversarial region
            loss = adv_logit[torch.arange(batch_size), target] - adv_logit[torch.arange(batch_size), label]
            grad = torch.autograd.grad(loss.mean() * batch_size, adv_image, create_graph=training)[0]
            adv_image.requires_grad = False

            # resize grad to the scale of normal_mean
            if grad.shape[-1] != normal_mean.shape[-1]:
                grad = F.interpolate(grad, size=normal_mean.shape[-1], mode='bilinear', align_corners=True)

            # add normal mean
            grad = grad + normal_mean.view(1, *normal_mean.shape).repeat(batch_size, 1, 1, 1)

            # calibration: tune l2 norm of output then mix grad and empty_normal_mean using empty_coeff
            if calibrate:
                grad = grad / torch.clamp(grad.view(batch_size, -1).norm(dim=1).view(-1, 1, 1, 1), min=1e-2)
                empty_normal_mean = empty_normal_mean / torch.clamp(empty_normal_mean.norm(), min=1e-2)
                empty_coeff.data[:] = torch.clamp(empty_coeff.data, 0, 1)
                grad = (1. - empty_coeff) * grad + \
                       empty_coeff * empty_normal_mean.view(1, *empty_normal_mean.shape).repeat(batch_size, 1, 1, 1)
            output['grad'] = grad
    if 'logit' in output_fields:
        output['logit'] = get_logit(image)
    return output
