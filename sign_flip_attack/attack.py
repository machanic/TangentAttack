import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
import random
import json
from collections import defaultdict, OrderedDict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np

from config import IMAGE_DATA_ROOT, CLASS_NUM, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
import glog as log
import os.path as osp
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, TinyImageNetDataset


class SignFlipAttack(object):
    def __init__(self, model, dataset, clip_min, clip_max, epsilon, targeted, batch_size, resize_factor=1., max_queries=1000):
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon
        self.resize_factor = resize_factor
        assert self.resize_factor >= 1.
        self.targeted = targeted
        self.maximum_queries = max_queries

        self.ord = np.inf
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.dataset = dataset
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def get_image_of_target_class(self,dataset_name, target_labels, target_model):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "TinyImageNet":
                dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
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

    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):

        dist = torch.norm((perturbed - images).view(images.size(0), -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                inside_batch_index].item()

    def resize(self, x, h, w):
        return F.interpolate(x, size=[h, w], mode='bilinear', align_corners=False)

    def project_infinity(self, x_a, x, l):
        '''
        linf projection
        '''
        return torch.max(x - l[:, None, None, None], torch.min(x_a, x + l[:, None, None, None]))

    def get_predict_label(self, x):
        return self.model(x.cuda()).cpu().argmax(1)

    def is_adversarial(self, x, y, targeted):
        '''
        check whether the adversarial constrain holds for x
        '''
        if targeted:
            return self.get_predict_label(x) == y
        else:
            return self.get_predict_label(x) != y

    def binary_infinity(self, x_a, x, y, k):
        '''
        linf binary search
        :param k: the number of binary search iteration
        '''
        b = x_a.size(0)
        l = torch.zeros(b)
        u, _ = (x_a - x).view(b, -1).abs().max(1)
        for _ in range(k):
            mid = (l + u) / 2
            adv = self.project_infinity(x_a, x, mid).clamp(self.clip_min, self.clip_max)
            check = self.is_adversarial(adv, y, self.targeted)
            u[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
            check = check < 1
            l[check.nonzero().squeeze(1)] = mid[check.nonzero().squeeze(1)]
        return self.project_infinity(x_a, x, u).clamp(self.clip_min, self.clip_max)

    def attack(self, batch_index, x, x_a, y):
        '''
        Sign Flip Attack: linf decision-based adversarial attack
        :param batch_index: current batch index of all images
        :param x: original images, torch tensor of size (b,c,h,w)
        :param x_a: initial images for targeted attacks, torch tensor of size (b,c,h,w). None for untargeted attacks
        :param y: original labels for untargeted attacks, target labels for targeted attacks, torch tensor of size (b,)
        :param resize_factor: dimensionality reduction rate, >= 1.0
        :param targeted: attack mode, True for targeted attacks, False for untargeted attacks
        :param max_queries: maximum query number
        :param linf: linf threshold
        :return: adversarial examples and corresponding required queries
        '''
        # initialize
        if self.targeted:
            assert x_a is not None
            check = self.is_adversarial(x_a, y, self.targeted)
            if check.sum().item() < y.size(0):
                log.info('Some initial images do not belong to the target class!')
                # return x, torch.zeros(x.size(0))
            check = self.is_adversarial(x, y, self.targeted)
            if check.sum().item() > 0:
                log.info('Some original images already belong to the target class!')
                # return x, torch.zeros(x.size(0))
        else:
            check = self.is_adversarial(x, y, True)
            if check.sum().item() < y.size(0):
                log.info('Some original images do not belong to the original class!')
                # return x, torch.zeros(x.size(0))
            x_a = torch.rand_like(x)
            iters = 0
            check = self.is_adversarial(x_a, y, self.targeted)
            while check.sum().item() < y.size(0):
                x_a[check < 1] = torch.rand_like(x_a[check < 1])
                check = self.is_adversarial(x_a, y, self.targeted)
                iters += 1
                if iters > 10000:
                    log.info('Initialization by random noise Failed! Use randomly selected a image with randomly selected class')
                    target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                  size=y.size()).long()
                    invalid_target_index = target_labels.eq(y)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                            size=target_labels[invalid_target_index].size()).long()
                        invalid_target_index = target_labels.eq(y)

                    x_a = self.get_image_of_target_class(self.dataset, target_labels, self.model)
                    break

        # linf binary search
        x_a = self.binary_infinity(x_a, x, y, 10)
        delta = x_a - x

        b, c, h, w = delta.size()
        batch_image_positions = np.arange(batch_index * b,
                                          min((batch_index + 1) * b, self.total_images)).tolist()

        h_dr, w_dr = int(h // self.resize_factor), int(w // self.resize_factor)

        # query: query number for each image, it will not stop counting query when distortion < args.epsilon
        query = torch.zeros(b)
        success_stop_queries = torch.zeros(b)

        # q_num: current queries
        q_num = 0
        # 10 queries for binary search
        q_num, query, success_stop_queries = q_num + 10, query + 10, success_stop_queries+10

        dist = torch.norm((x_a - x).view(b, -1), self.ord, 1)
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        # indices for unsuccessful images
        unsuccessful_indices = torch.ones(b,dtype=torch.bool)

        # hyper-parameter initialization
        alpha = torch.ones(b) * 0.004
        prob = torch.ones_like(delta) * 0.999
        prob = self.resize(prob, h_dr, w_dr)

        # additional counters for hyper-parameter adjustment
        reset = 0
        proj_success_rate = torch.zeros(b)
        flip_success_rate = torch.zeros(b)

        while q_num < self.maximum_queries:
            reset += 1
            b_cur = unsuccessful_indices.sum()

            # the project step
            eta = torch.randn([b_cur, c, h_dr, w_dr]).sign() * alpha[unsuccessful_indices][:, None, None, None]
            eta = self.resize(eta, h, w)
            l, _ = delta[unsuccessful_indices].abs().view(b_cur, -1).max(1)
            delta_p = self.project_infinity(delta[unsuccessful_indices] + eta, torch.zeros_like(eta), l - alpha[unsuccessful_indices])
            check = self.is_adversarial((x[unsuccessful_indices] + delta_p).clamp(self.clip_min, self.clip_max), y[unsuccessful_indices],
                                   self.targeted)
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_p[
                check.nonzero().squeeze(1)]
            proj_success_rate[unsuccessful_indices] += check.float()
            query[unsuccessful_indices] += 1

            self.count_stop_query_and_distortion(x,torch.clamp(x + delta,min=self.clip_min,max=self.clip_max), query,
                                                 success_stop_queries, batch_image_positions)


            # the random sign flip step
            s = torch.bernoulli(prob[unsuccessful_indices]) * 2 - 1
            delta_s = delta[unsuccessful_indices] * self.resize(s, h, w).sign()
            check = self.is_adversarial((x[unsuccessful_indices] + delta_s).clamp(self.clip_min, self.clip_max), y[unsuccessful_indices],
                                   self.targeted)
            prob[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] -= s[check.nonzero().squeeze(
                1)] * 1e-4
            prob.clamp_(0.99, 0.9999)
            flip_success_rate[unsuccessful_indices] += check.float()
            delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_s[
                check.nonzero().squeeze(1)]
            query[unsuccessful_indices] += 1
            self.count_stop_query_and_distortion(x, torch.clamp(x + delta,min=self.clip_min,max=self.clip_max), query,
                                                 success_stop_queries, batch_image_positions)

            # hyper-parameter adjustment
            if reset % 10 == 0:
                proj_success_rate /= reset
                flip_success_rate /= reset
                alpha[proj_success_rate > 0.7] *= 1.5
                alpha[proj_success_rate < 0.3] /= 1.5
                prob[flip_success_rate > 0.7] -= 0.001
                prob[flip_success_rate < 0.3] += 0.001
                prob.clamp_(0.99, 0.9999)
                reset *= 0
                proj_success_rate *= 0
                flip_success_rate *= 0

            # query count
            q_num += 2


            # update indices for unsuccessful perturbations
            l, _ = delta[unsuccessful_indices].abs().view(b_cur, -1).max(1)
            # FIXME 下面一句话要改，因为一旦epsilon小于指定值，Q停止记录数字了
            # unsuccessful_indices[unsuccessful_indices.nonzero().squeeze(1)[(l <= self.epsilon).nonzero().squeeze(1)]] = 0

            # print attack information
            if q_num % 10000 == 0:
                log.info(f"Queries: {q_num}/{self.maximum_queries} Successfully attacked images: {b - unsuccessful_indices.sum()}/{b}")

            # if unsuccessful_indices.sum() == 0:
            #     break

        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        dist = torch.norm(((x + delta).clamp(self.clip_min, self.clip_max) - x).view(b, -1), self.ord, 1)
        log.info(f"Finished {batch_index}-th batch! Queries: {q_num}/{self.maximum_queries} Successfully attacked images: {(dist <= self.epsilon).int().sum()}/{b}")
        return (x + delta).clamp(self.clip_min, self.clip_max), query, success_stop_queries, dist, dist <= self.epsilon



    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and target_model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = target_model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
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

                target_images = self.get_image_of_target_class(self.dataset,target_labels, target_model)
                labels = target_labels
            else:
                target_labels = None
                target_images = None
                labels = true_labels

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index,
                                                                                    images, target_images, labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            with torch.no_grad():
                adv_logit = target_model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.float() *(success_query <= self.maximum_queries).float()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.bool()].median().item(),
                          "max_query": self.success_query_all[self.success_all.bool()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_all":self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'sign_flip_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'sign_flip_attack-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/SFA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float,  help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, default='linf')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
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
    parser.add_argument('--resize_factor',type=float,default=2.0)

    args = parser.parse_args()
    assert args.norm == "linf"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args_dict = None
    args.epsilon = json.load(open(args.json_config))[args.dataset]["epsilon"]
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
        attacker = SignFlipAttack(model, args.dataset, 0, 1.0, args.epsilon, args.targeted, args.batch_size, args.resize_factor,
                                  args.max_queries)
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()





