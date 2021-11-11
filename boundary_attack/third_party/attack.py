import argparse
import sys
import os
sys.path.append(os.getcwd())
import json
import random

from collections import OrderedDict, defaultdict
from types import SimpleNamespace
import numpy as np
import os
import torch
from torch.nn import functional as F
import glog as log

from config import CLASS_NUM, IMAGE_DATA_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
import os.path as osp


class BoundaryAttack(object):
    def __init__(self, model, dataset, norm, epsilon,
                 batch_size, targeted, clip_min, clip_max, maximum_queries):
        self.model = model
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.image_shape = self.model.input_size
        self.targeted = targeted  # True if targeted attack.
        self.epsilon = epsilon
        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_best_all = torch.zeros_like(self.query_all)

    def orthogonal_perturbation(self, delta, prev_sample, target_sample):
        """Generate orthogonal perturbation."""
        perturb = torch.randn(1, self.image_shape[0], self.image_shape[1], self.image_shape[2]).to(target_sample.device)
        perturb /= torch.linalg.norm(perturb, dim=(2,3),keepdim=True)
        perturb *= delta * torch.mean(self.get_diff(target_sample, prev_sample))
        # Project perturbation onto sphere around target
        diff = (target_sample - prev_sample).type(torch.float32)  # Orthorgonal vector to sphere surface
        diff /= self.get_diff(target_sample, prev_sample)  # Orthogonal unit vector
        # We project onto the orthogonal then subtract from perturb
        # to get projection onto sphere surface
        perturb -= (torch.vdot(perturb.view(-1), diff.view(-1)) / torch.square(torch.linalg.norm(diff))) * diff
        # Check overflow and underflow
        mean_val = torch.zeros(self.image_shape[0]).view(1,self.image_shape[0],1,1).to(perturb.device)
        overflow = (prev_sample + perturb) - 1.0 + mean_val  # B,C,H,W - C
        perturb -= overflow * (overflow > 0)
        underflow = -mean_val
        perturb += underflow * (underflow > 0)
        return perturb

    def get_diff(self, sample_1, sample_2):
        """Channel-wise norm of difference between samples."""
        # B,C,H,W
        return torch.linalg.norm(sample_1 - sample_2, dim=(2,3),keepdim=True)

    def forward_perturbation(self, epsilon, prev_sample, target_sample):
        """Generate forward perturbation."""
        perturb = (target_sample - prev_sample).type(torch.float32)
        perturb *= epsilon
        return perturb

    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):

        dist = torch.norm((perturbed - images).view(1, -1), self.ord, 1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                inside_batch_index].item()
        return dist

    def get_image_of_target_class(self,dataset_name, target_labels, target_model):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name=="CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")

            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
            while logits.max(1)[1].item() != label.item():
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
                with torch.no_grad():
                    logits = target_model(image.cuda())
            assert true_label == label.item()
            images.append(torch.squeeze(image))
        return torch.stack(images) # B,C,H,W

    def attack(self, image_index, images, target_images, true_label, target_label):
        # 他的代码里的target_sample和target_class才是原始图片分类
        assert images.dim()==4
        assert target_images.dim()==4
        batch_image_positions = np.arange(image_index, min(image_index + 1, self.total_images)).tolist()
        adv_images = target_images.clone()
        n_steps = 0
        epsilon = 1.
        delta = 0.1
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        # Move first step to the boundary
        if self.targeted:
            label = target_label
        else:
            label = true_label
        while True:
            trial_sample = adv_images + self.forward_perturbation(epsilon, adv_images, images)
            query+=1
            predict_label = self.model(trial_sample).max(1)[1].item()

            if (predict_label == label)==self.targeted:
                adv_images = trial_sample  # 仍然保持攻击成功的状态
                self.count_stop_query_and_distortion(images,adv_images,query,success_stop_queries,batch_image_positions)
                break
            else:
                epsilon *= 0.9
        # Iteratively run attack
        while True:
            # log.info("Step {}...".format(n_steps))
            # Orthogonal step
            # log.info("Delta step...")
            d_step = 0
            while True:
                d_step += 1
                trial_samples = []
                for i in np.arange(10):
                    trial_sample = adv_images + self.orthogonal_perturbation(delta, adv_images, images)
                    trial_samples.append(trial_sample)
                trial_samples = torch.cat(trial_samples, 0)
                query += trial_samples.size(0)
                predict_labels = self.model(trial_samples).max(1)[1]
                if self.targeted:
                    d_score = torch.mean((predict_labels==target_label).float())
                else:
                    d_score = torch.mean((predict_labels!=true_label).float())

                if d_score > 0.0:
                    if d_score < 0.3:
                        delta *= 0.9
                    elif d_score > 0.7:
                        delta /= 0.9
                    if self.targeted:
                        indexes = predict_labels==target_label
                    else:
                        indexes = predict_labels!=true_label
                    adv_images = trial_samples[indexes][0].unsqueeze(0)
                    dist = self.count_stop_query_and_distortion(images, adv_images, query, success_stop_queries,
                                                         batch_image_positions)
                    log.info(
                        "{}-th image, distortion:{:.4f} query:{}".format(image_index + 1, dist.item(),
                                                                         query[0].item()))
                    break
                else:
                    delta *= 0.9
            # Forward step
            # log.info("Epsilon step...")
            e_step = 0
            while True:
                e_step += 1
                trial_sample = adv_images + self.forward_perturbation(epsilon, adv_images, images)
                predict_label = self.model(trial_sample).max(1)[1].item()
                query += trial_sample.size(0)
                assert trial_sample.size(0)==1
                if (predict_label == label) == self.targeted:
                    adv_images = trial_sample
                    epsilon /= 0.5
                    dist = self.count_stop_query_and_distortion(images, adv_images, query, success_stop_queries,
                                                         batch_image_positions)
                    log.info(
                        "{}-th image, distortion:{:.4f} query:{}".format(image_index + 1, dist.item(),
                                                                         query[0].item()))
                    break
                elif e_step > 500:
                    break
                else:
                    epsilon *= 0.5

            n_steps += 1

            if query[0].item() >= self.maximum_queries:
                break
        return adv_images, query, success_stop_queries, dist, (dist <= self.epsilon)

    def find_closest_img(self, X_orig, X_targets, label, is_targeted):
        """
        From a list of potential starting images, finds the closest to the original.
        Before returning, this method makes sure that the image fulfills the adversarial condition (is actually classified as the target label).
        :param X_orig: The original image to attack.
        :param X_targets: List of images that fulfill the adversarial criterion (i.e. target class in the targeted case)
        :param is_targeted: true if the attack is targeted.
        :param label: the target label if targeted, or the correct label if untargeted.
        :return: the closest image (in L2 distance) to the original that also fulfills the adversarial condition.
        """

        dists = torch.empty(X_targets.size(0), dtype=torch.float32).to(X_targets.device)
        for i in range(X_targets.size(0)):
            d_l2 = torch.linalg.norm(X_targets[i] - X_orig)
            dists[i] = d_l2

        indices = torch.argsort(dists)
        for index in indices:
            X_target = X_targets[index]
            pred_clsid = self.model(X_target).max(1)[1].item()
            if (pred_clsid == label) == is_targeted:
                log.info("Found an image of the target class, d_l2={:.3f}.".format(dists[index]))
                return X_target

            log.info("Image of target class is wrongly classified by model, skipping.")

        return X_targets[random.randint(0,X_targets.size(0)-1)]

    def attack_all_images(self, args, arch_name, result_dump_path):

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            images = images.cuda()
            with torch.no_grad():
                logit = self.model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0: # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index+1))
                continue
            selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[invalid_target_index].shape).long()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = self.get_image_of_target_class(self.dataset_name,target_labels, self.model)

                label = target_labels[0].item()
            else:
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                              size=true_labels.size()).long()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long()
                    invalid_target_index = target_labels.eq(true_labels)
                target_images = self.get_image_of_target_class(self.dataset_name, target_labels, self.model)
                label = true_labels[0].item()
            target_images = target_images.cuda()
            # target_images = self.find_closest_img(images, target_images, label, self.targeted)
            adv_images, query, success_query, distortion_best, success_epsilon = self.attack(batch_index, images, target_images,
                                                                          true_labels[0].item(),target_labels[0].item())
            distortion_best = distortion_best.detach().cpu()
            with torch.no_grad():
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * success_epsilon.float().detach().cpu()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_best"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          # "mean_query": self.success_query_all[self.success_all.bool()].mean().item(),
                          # "median_query": self.success_query_all[self.success_all.bool()].median().item(),
                          # "max_query": self.success_query_all[self.success_all.bool()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_all":self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_best": self.distortion_best_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))



def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'boundary_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'boundary_attack-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config', type=str, default='./configures/BBA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--max_queries',type=int, default=10000)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    assert args.batch_size == 1, "BBA only supports mini-batch size equals 1!"
    assert args.norm == "l2", "Please modify line_search_to_boundary function in Linf norm attack"
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 20000
    assert args.targeted, "This code only supports targeted attack"
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type,
                                             args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = BoundaryAttack(model, args.dataset, args.norm, args.epsilon, args.batch_size,
                                   args.targeted, 0.0, 1.0, args.max_queries)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
