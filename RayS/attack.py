import argparse
import copy
import json
import os
import glob
import sys
from collections import defaultdict, OrderedDict

sys.path.append(os.getcwd())
from types import SimpleNamespace
from torch.nn import functional as F
import numpy as np
import random
import torch
import glog as log
from dataset.dataset_loader_maker import DataLoaderMaker
from config import CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT, IN_CHANNELS, IMAGE_SIZE
import os.path as osp
import os
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel

class RayS(object):
    def __init__(self, model, dim, clip_min, clip_max, epsilon_l2, epsilon_linf, dataset, batch_size, maximum_queries):
        self.model = model
        self.ord_l2 = 2
        self.ord_linf = np.inf
        self.epsilon_l2 = epsilon_l2
        self.epsilon_linf = epsilon_linf
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.batch_size = batch_size
        self.dim = dim
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.maximum_queries = maximum_queries
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.dataset_loader.dataset)

        self.query_all = torch.zeros(self.total_images)
        self.distortion_l2_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.distortion_linf_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images

        self.not_done_l2_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_l2_all = torch.zeros_like(self.query_all)
        self.success_query_l2_all = torch.zeros_like(self.query_all)

        self.not_done_linf_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_linf_all = torch.zeros_like(self.query_all)
        self.success_query_linf_all = torch.zeros_like(self.query_all)

        self.l2_distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.linf_distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def get_xadv(self, x, v, d):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(x.size(0)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v  # TODO explain the first iteration: inifity * v ??
        out = torch.clamp(out, self.clip_min, self.clip_max)
        return out


    def attack(self, batch_index, images, true_labels, target_labels=None, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(images.size())
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                min((batch_index + 1) * self.batch_size, self.total_images)).tolist()
        # init variables
        query = torch.zeros_like(true_labels).float()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(true_labels).float().fill_(float("Inf")).cuda()  # 就是半径r_t
        working_ind_of_l2_norm = (self.d_t > self.epsilon_l2).nonzero().flatten()  # 大于epsilon的才继续寻找，否则都已经成功了
        working_ind_of_linf_norm = (self.d_t > self.epsilon_linf).nonzero().flatten()   # 大于epsilon的才继续寻找，否则都已经成功了
        success_stop_queries_of_l2_norm = query.clone()  # stop query count once the distortion < epsilon
        success_stop_queries_of_linf_norm = query.clone()
        dist_l2 = self.d_t.clone()
        dist_linf = self.d_t.clone()
        self.x_final = self.get_xadv(images, self.sgn_t, self.d_t)

        block_level = 0
        block_ind = 0
        for i in range(self.maximum_queries):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))  # np.ceil: the minimum integer value that greater or equal than this float value
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (query < self.maximum_queries)
            attempt = self.sgn_t.clone().view(shape[0], self.dim)  # 注意是clone过的，意思是实验一下这个方向，如果好了（半径更小），才更新sgn_t
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1. # the block is negative
            attempt = attempt.view(shape)

            self.binary_search(images, true_labels, target_labels, query, attempt, valid_mask)  # attempt is the direction that we explore

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist_l2 = torch.norm((self.x_final - images).view(shape[0], -1), self.ord_l2, 1)
            dist_linf = torch.norm((self.x_final - images).view(shape[0], -1), self.ord_linf, 1)
            success_stop_queries_of_l2_norm[working_ind_of_l2_norm] = query[working_ind_of_l2_norm]
            success_stop_queries_of_linf_norm[working_ind_of_linf_norm] = query[working_ind_of_linf_norm]
            working_ind_of_l2_norm = (dist_l2 > self.epsilon_l2).nonzero().flatten()
            working_ind_of_linf_norm = (dist_linf > self.epsilon_linf).nonzero().flatten()

            for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                self.distortion_l2_all[index_over_all_images][query[inside_batch_index].item()] = dist_l2[inside_batch_index].item()
                self.distortion_linf_all[index_over_all_images][query[inside_batch_index].item()] = dist_linf[
                    inside_batch_index].item()
            if torch.sum(query >= self.maximum_queries) == shape[0]:
                log.info('out of {} queries'.format(self.maximum_queries))
                break
            log.info('Attacking image {} - {} / {}, d_t(avg radius): {:.4f} | avg L2 dist: {:.4f} | avg Linf dist: {:.4f}  | avg queries: {:.4f} | min query: {} | L2 rob acc: {:.4f} | Linf rob acc: {:.4f}'.format(
                batch_index * self.batch_size, (batch_index + 1) * self.batch_size, self.total_images,
                torch.mean(self.d_t), torch.mean(dist_l2),torch.mean(dist_linf), torch.mean(query.float()), int(query.min().item()),
                len(working_ind_of_l2_norm) / len(images), len(working_ind_of_linf_norm) / len(images)))


        success_stop_queries_of_l2_norm = torch.clamp(success_stop_queries_of_l2_norm, 0, self.maximum_queries)
        success_stop_queries_of_linf_norm = torch.clamp(success_stop_queries_of_linf_norm, 0, self.maximum_queries)
        # adv_images, query, success_query_l2, success_query_linf, l2_distortion_with_max_queries, linf_distortion_with_max_queries, success_epsilon_l2, success_epsilon_linf
        return self.x_final, query, success_stop_queries_of_l2_norm,success_stop_queries_of_linf_norm , dist_l2, dist_linf, \
                    dist_l2 <= self.epsilon_l2, dist_linf <= self.epsilon_linf

    # check whether solution is found
    def search_succ(self,  x, y, target, mask, query):
        query[mask] += 1  # 攻击成功(label发生改变到指定label)，query不再增加，除此之外，query超过max_queries的也继续攻击
        if target is not None:
            return self.model.forward(x[mask]).max(1)[1] == target[mask]
        else:
            return self.model.forward(x[mask]).max(1)[1] != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, images, true_labels, target_labels, query, sgn, valid_mask, tol=1e-3):
        # valid_mask is those images with queries less than maximum query number
        sgn_norm = torch.norm(sgn.view(len(images), -1), p=2, dim=1)
        sgn_unit = sgn / sgn_norm.view(len(images), 1, 1, 1)

        d_start = torch.zeros_like(true_labels).float().cuda()
        d_end = self.d_t.clone()
        # initial_succ_mask is a boolean mask
        initial_succ_mask = self.search_succ(self.get_xadv(images, sgn_unit, self.d_t), true_labels, target_labels, valid_mask, query)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]  # 所有valid的mask，进一步找出success的
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]  # self.d_t is r_best of the paper, which is sed as infinity

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            # binary search, test the middle 注意这里query超过max_query的也继续增加新的query，继续攻击
            search_succ_mask = self.search_succ(self.get_xadv(images, sgn_unit, d_mid), true_labels, target_labels, to_search_ind, query)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]  # d_end - d_start > 1e-3

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:  #  refer to Algorithm 1  line 9~10
            self.d_t[to_update_ind] = d_end[to_update_ind] # update r_best, sgn这个方向的更小的半径
            self.x_final[to_update_ind] = self.get_xadv(images, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]  # 只有获得更小的半径才更新方向sgn_t, refer to Algorithm 1  line 9~10

    def __call__(self, data, label, target=None, query_limit=10000):
        return self.attack(data, label, target)

    def attack_all_images(self, args, arch_name, target_model, l2_norm_result_dump_path, linf_norm_result_dump_path):
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and target_model.input_size[-1] != 299:
                images = F.interpolate(images, size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear', align_corners=False)
            images = images.cuda()
            true_labels = true_labels.cuda()
            with torch.no_grad():
                logit = target_model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels).float()  # shape = (batch_size,)
            selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            # x_final, query,success_stop_queries_of_l2_norm,success_stop_queries_of_linf_norm , dist_l2, dist_linf, (dist_l2 <= self.epsilon_l2), dist_linf <= self.epsilon_linf
            adv_images, query, success_query_l2, success_query_linf, l2_distortion_with_max_queries, linf_distortion_with_max_queries, success_epsilon_l2, success_epsilon_linf = self.attack(batch_index, images.cuda(), true_labels.cuda(), target_labels)
            l2_distortion_with_max_queries = l2_distortion_with_max_queries.detach().cpu()
            linf_distortion_with_max_queries = linf_distortion_with_max_queries.detach().cpu()
            success_l2 = success_epsilon_l2.float()  # query超过max_queries，只要仍然在分类边界的target一侧，就继续增加query查询，因此即使攻击成功，也可能query > epsilon
            success_linf = success_epsilon_linf.float()
            not_done_l2 = torch.ones_like(success_l2) - success_l2
            not_done_linf = torch.ones_like(success_linf) - success_linf
            success_l2 = success_l2 * correct
            success_linf = success_linf * correct

            for key in ['query', 'correct', 'not_done_l2', "not_done_linf",
                        'success_l2', 'success_linf', 'success_query_l2', 'success_query_linf', "l2_distortion_with_max_queries", "linf_distortion_with_max_queries"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))

        log.info('Saving results to {}'.format(l2_norm_result_dump_path))
        args_l2 = copy.deepcopy(vars(args))
        args_l2["norm"] = "l2"
        args_l2["epsilon"] = self.epsilon_l2
        meta_info_dict_l2_norm = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_l2_all[self.correct_all.bool()].mean().item(),
                          # "mean_query": self.success_query_l2_all[self.success_l2_all.bool()].mean().item(),
                          # "median_query": self.success_query_l2_all[self.success_l2_all.bool()].median().item(),
                          # "max_query": self.success_query_l2_all[self.success_l2_all.bool()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_l2_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                      "success_query_all": self.success_query_l2_all.detach().cpu().numpy().astype(
                                  np.int32).tolist(),
                                  "success_all": self.success_l2_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_l2_all,
                          "avg_distortion_with_max_queries": self.l2_distortion_with_max_queries_all.mean().item(),
                          "args": args_l2}
        with open(l2_norm_result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict_l2_norm, result_file_obj, sort_keys=True)
        log.info("L2 norm result is written to {}".format(l2_norm_result_dump_path))

        args_linf = copy.deepcopy(vars(args))
        args_linf["norm"] = "linf"
        args_linf["epsilon"] = self.epsilon_linf
        meta_info_dict_linf_norm = {"avg_correct": self.correct_all.mean().item(),
                                  "avg_not_done": self.not_done_linf_all[self.correct_all.bool()].mean().item(),
                                  # "mean_query": self.success_query_linf_all[self.success_linf_all.bool()].mean().item(),
                                  # "median_query": self.success_query_linf_all[self.success_linf_all.bool()].median().item(),
                                  # "max_query": self.success_query_linf_all[self.success_linf_all.bool()].max().item(),
                                  "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                                  "not_done_all": self.not_done_linf_all.detach().cpu().numpy().astype(np.int32).tolist(),
                                  "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                                    "success_query_all": self.success_query_linf_all.detach().cpu().numpy().astype(
                                        np.int32).tolist(),
                                    "success_all": self.success_linf_all.detach().cpu().numpy().astype(np.int32).tolist(),
                                  "distortion": self.distortion_linf_all,
                                  "avg_distortion_with_max_queries": self.linf_distortion_with_max_queries_all.mean().item(),
                                  "args": args_linf}
        with open(linf_norm_result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict_linf_norm, result_file_obj, sort_keys=True)
        log.info("Linf norm result is written to {}".format(linf_norm_result_dump_path))

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'RayS_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'RayS-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args, epsilon_l2, epsilon_linf):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))
    log.info('{:s}: {}'.format("epsilon_l2", epsilon_l2))
    log.info('{:s}: {}'.format("epsilon_linf", epsilon_linf))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config', type=str, default='./configures/RayS.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size.')
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
    parser.add_argument('--max_queries',type=int,default=10000)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    args_dict = None
    # If a json file is given, use the JSON file as the base, and then update it with args
    defaults = json.load(open(args.json_config))[args.dataset]
    epsilon_l2 = defaults["l2"]["epsilon"]
    epsilon_linf = defaults["linf"]["epsilon"]
    if args.targeted:
        if args.dataset == "ImageNet":
           args.max_queries = 20000
    exp_dir_l2_norm = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, "l2", args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    exp_dir_linf_norm = osp.join(args.exp_dir,
                               get_exp_dir_name(args.dataset, "linf", args.targeted, args.target_type,
                                                args))  # 随机产生一个目录用于实验
    os.makedirs(exp_dir_l2_norm, exist_ok=True)
    os.makedirs(exp_dir_linf_norm, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(exp_dir_l2_norm, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(exp_dir_l2_norm, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(exp_dir_l2_norm, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(exp_dir_l2_norm, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.all_archs:
        archs = []
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args,epsilon_l2,epsilon_linf)

    for arch in archs:
        if args.attack_defense:
            save_result_path_l2_norm = exp_dir_l2_norm + "/{}_{}_result.json".format(arch, args.defense_model)
            save_result_path_linf_norm = exp_dir_linf_norm + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path_l2_norm = exp_dir_l2_norm + "/{}_result.json".format(arch)
            save_result_path_linf_norm = exp_dir_linf_norm + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path_l2_norm) and os.path.exists(save_result_path_linf_norm):
            continue
        log.info("Begin attack {} on {}, result will be saved to {} and {}".format(arch, args.dataset, save_result_path_l2_norm, save_result_path_linf_norm))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = RayS(model, dim=IN_CHANNELS[args.dataset] * model.input_size[1] * model.input_size[2],
                        clip_min=0, clip_max=1.,epsilon_l2=epsilon_l2, epsilon_linf=epsilon_linf,
                        dataset=args.dataset, batch_size=args.batch_size, maximum_queries=args.max_queries)
        attacker.attack_all_images(args, arch, model, save_result_path_l2_norm, save_result_path_linf_norm)
        model.cpu()
