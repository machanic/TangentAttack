#!/usr/bin/env python3
import sys
import os
from types import SimpleNamespace

sys.path.append(os.getcwd())

import os.path as osp
import argparse
import json
import random
from collections import OrderedDict, defaultdict
import socket
import getpass

import torch
import numpy as np

from config import CLASS_NUM, IMAGE_DATA_ROOT, MODELS_TEST_STANDARD
from dataset.target_class_dataset import CIFAR10Dataset, CIFAR100Dataset, ImageNetDataset,TinyImageNetDataset
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
import glog as log
from torch.nn import functional as F
import boundary_attack.foolbox as foolbox
from boundary_attack.foolbox.attacks.boundary_attack import BoundaryAttack

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--dataset', required=True, type=str, choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', 'TinyImageNet'],
                        help='which dataset to use')
    parser.add_argument('--phase', default='test', type=str, choices=['train', 'val', 'valv2', 'test'],
                        help='train, val, test')
    parser.add_argument('--arch', default=None, type=str,
                        help='victim network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--norm', default='l2', type=str, choices=['l2'],
                        help='l2 attack or linf attack')
    parser.add_argument('--attack-method', default='ba', choices=['ba', 'cw', 'bapp'],
                        help='attack method')
    parser.add_argument('--save-all-steps', action='store_true',
                        help='save all intermediate adversarial images')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    parser.add_argument('--json-config', type=str, default='./configures/boundary_attack.json',
                        help='a configures file to be passed in instead of arguments')
    # bapp (a.k.a., hsja) parameters
    parser.add_argument('--bapp-iteration', default=132, type=int,
                        help='boundary attack++: number of iterations')
    parser.add_argument('--bapp-initial-num-eval', default=100, type=int,
                        help='boundary attack++: initial number of evaluations for gradient estimation')
    parser.add_argument('--bapp-max-num-eval', default=10000, type=int,
                        help='boundary attack++: max number of evaluations for gradient estimation')
    parser.add_argument('--bapp-stepsize-search', default='geometric_progression', type=str,
                        choices=['geometric_progression', 'grid_search'],
                        help='boundary attack++: step size search method')
    parser.add_argument('--bapp-gamma', default=0.01, type=float,
                        help='boundary attack++: to decide binary search threshold')
    parser.add_argument('--bapp-batch-size', default=256, type=int,
                        help='boundary attack++: batch size for model prediction')
    parser.add_argument('--bapp-internal-dtype', default='float32', type=str,
                        help='boundary attack++: internal dtype. foolbox default value is float64')

    # boundary attack parameters
    parser.add_argument('--ba-iteration', default=1200, type=int,
                        help='boundary attack: number of iterations')
    parser.add_argument('--ba-max-directions', default=25, type=int,
                        help='boundary attack: batch size')
    parser.add_argument('--ba-spherical-step', default=1e-2, type=float,
                        help='boundary attack: spherical step size')
    parser.add_argument('--ba-source-step', default=1e-2, type=float,
                        help='boundary attack: source step size')
    parser.add_argument('--ba-step-adaptation', default=1.5, type=float,
                        help='boundary attack: step size adaptation multiplier')
    parser.add_argument('--ba-batch-size', default=1, type=int,
                        help='boundary attack: batch size')
    parser.add_argument('--ba-no-tune-batch-size', action='store_true',
                        help='boundary attack: disable automatic batch size tuning')
    parser.add_argument('--ba-no-threaded', action='store_true',
                        help='boundary attack: do not use multi thread to generate candidate and random numbers')
    parser.add_argument('--ba-internal-dtype', default='float32', type=str,
                        help='boundary attack: internal dtype. foolbox default value is float64')

    # cw attack (white-box) parameters
    parser.add_argument('--cw-binary-search-step', default=5, type=int,
                        help='cw attack: number of binary search steps of constant')
    parser.add_argument('--cw-max-iteration', default=1000, type=int,
                        help='cw attack: maximum number of iterations')
    parser.add_argument('--cw-confidence', default=0.0, type=float,
                        help='cw attack: confidence threshold')
    parser.add_argument('--cw-learning-rate', default=0.005, type=float,
                        help='cw learning: initial learning rate')
    parser.add_argument('--cw-initial-const', default=0.01, type=float,
                        help='cw attack: initial constant')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str, default="")
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_image_of_target_class(dataset_name, target_labels, target_model):

    images = []
    for label in target_labels:  # length of target_labels is 1
        if dataset_name == "ImageNet":
            dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
        elif dataset_name == "CIFAR-10":
            dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
        elif dataset_name=="CIFAR-100":
            dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
        elif dataset_name == "TinyImageNet":
            dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
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

def main(args, result_dump_path):
    # make model
    log.info('Initializing model {} on {}'.format(args.arch, args.dataset))
    if args.attack_defense:
        model = DefensiveModel(args.dataset, args.arch, no_grad=True, defense_model=args.defense_model,
                               norm=args.defense_norm, eps=args.defense_eps)
    else:
        model = StandardModel(args.dataset, args.arch, no_grad=True, load_pretrained=True)
    model.cuda()
    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=CLASS_NUM[args.dataset], device=str(args.device))
    log.info('Foolbox model created')
    distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
    result_json = {"statistical_details":{}}
    success_all = []
    # make loader
    loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.ba_batch_size)
    # extract image_id_ref from args.compare_with
    # args.compare_with should be the exp_dir of another experiment generated by policy_attack.py

    # these four variables represent type of visited images, and we treat them as boolean tensors
    # we use LongTensor instead of ByteTensor because we will do is_x.sum() later and ByteTensor will overflow
    correct_all = []
    is_ignore = torch.LongTensor(0)
    is_image_type = OrderedDict([
        ('train_seen', torch.LongTensor(0)),
        ('train_unseen', torch.LongTensor(0)),
        ('val', torch.LongTensor(0)),
        ('test', torch.LongTensor(0))
    ])
    # attack
    for batch_index, (image, label) in enumerate(loader):
        if args.dataset == "ImageNet" and model.input_size[-1] != 299:
            image = F.interpolate(image,
                                   size=(model.input_size[-2], model.input_size[-1]), mode='bilinear',
                                   align_corners=False)
        # extract inputs
        assert image.dim() == 4
        assert image.shape[0] == 1
        image = image.numpy()[0]
        # load init point and init query
        init_adv_image = init_distance = None
        if args.targeted:
            target_label = torch.fmod(label + 1, CLASS_NUM[args.dataset])
            init_adv_image = get_image_of_target_class(args.dataset, target_label, model).detach().cpu().numpy()[0]
        true_label = label.clone()
        label = label.item()
        pred = int(np.argmax(fmodel.forward_one(image)))

        # append 0, and we will modify them later
        is_ignore = torch.cat((is_ignore, torch.LongTensor([0])))

        # fill in is_correct, we will use is_correct to check is_ignore later
        correct_all.append(int(pred == label))


        # ignore image
        if is_ignore[-1].item():
            continue
        if int(correct_all[-1]) == 0:
            log.info("{}-th image is already incorrect classified, skip".format(batch_index))
            continue
        # start attack
        log.info('Begin attacking {}-th image'.format(batch_index))
        # initialize attack object and perform attack
        if not args.targeted:
            criterion = foolbox.criteria.Misclassification()
        else:
            criterion = foolbox.criteria.TargetClass((label + 1) % CLASS_NUM[args.dataset])
        if args.attack_method == 'ba':
            attack = BoundaryAttack(fmodel, criterion=criterion)
            with torch.no_grad():
                result = attack(input_or_adv=image,
                                label=label,
                                unpack=False,
                                iterations=args.ba_iteration,
                                max_directions=args.ba_max_directions,
                                max_queries=args.max_queries,
                                starting_point=init_adv_image,
                                initialization_attack=None,  # foolbox default
                                log_every_n_steps=100,
                                spherical_step=args.ba_spherical_step,
                                source_step=args.ba_source_step,
                                step_adaptation=args.ba_step_adaptation,
                                batch_size=args.ba_batch_size,
                                tune_batch_size=not args.ba_no_tune_batch_size,
                                threaded_rnd=not args.ba_no_threaded,
                                threaded_gen=not args.ba_no_threaded,
                                alternative_generator=False,  # foolbox default
                                internal_dtype=eval('np.{}'.format(args.ba_internal_dtype)),
                                save_all_steps=args.save_all_steps,
                                verbose=False)
        elif args.attack_method == 'cw':
            attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel, criterion=criterion)
            # cw attack does not required a starting point, since it starts from the clean image
            result = attack(input_or_adv=image,
                            label=label,
                            unpack=False,
                            binary_search_steps=args.cw_binary_search_step,
                            max_iterations=args.cw_max_iteration,
                            confidence=args.cw_confidence,
                            learning_rate=args.cw_learning_rate,
                            initial_const=args.cw_initial_const,
                            save_all_steps=args.save_all_steps,
                            abort_early=True)
        elif args.attack_method == 'bapp':
            attack = foolbox.attacks.BoundaryAttackPlusPlus(fmodel, criterion=criterion)
            with torch.no_grad():
                result = attack(input_or_adv=image,
                                label=label,
                                unpack=False,
                                iterations=args.bapp_iteration,
                                initial_num_evals=args.bapp_initial_num_eval,
                                max_num_evals=args.bapp_max_num_eval,
                                stepsize_search=args.bapp_stepsize_search,
                                gamma=args.bapp_gamma,
                                starting_point=init_adv_image,
                                batch_size=args.bapp_batch_size,
                                internal_dtype=eval('np.{}'.format(args.bapp_internal_dtype)),
                                log_every_n_steps=1,
                                save_all_steps=args.save_all_steps,
                                verbose=False)
        else:
            raise NotImplementedError('Unknown attack_method: {}'.format(args.attack_method))

        # attack current image done, print summary for current image
        if result.distance.value <= 0:
            log.info('Failed to attack {}-th image'.format(batch_index))
        log.info('Attack {}-th image done'.format(batch_index))
        log.info('  final query count: {}'.format(result._total_prediction_calls))
        log.info('     final distance: {:.4g} ({})'.format(result.distance.value, result.distance.name()))
        log.info('     final distance: {:.4f}'.format(np.sqrt(result.distance.value * image.size)))
        log.info('              label: {}'.format(label))
        log.info('               pred: {}'.format(pred))
        log.info('          adv label: {}'.format(result.adversarial_class))

        # save results
        log.info('Final result for {}-th image:  query: {:d}, dist: {:.4f}'.format(
            batch_index, result._total_prediction_calls, np.sqrt(result.distance.value * image.size)))
        if not hasattr(attack, "stats_distances"):
            log.info("Blend random noise failed! skip this {}-th image".format(batch_index))
            continue
        with torch.no_grad():
            adv_images = torch.from_numpy(result.perturbed)
            if adv_images.dim() == 3:
                adv_images = adv_images.unsqueeze(0)
            adv_logit = model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            if args.targeted:
                not_done = 1 - adv_pred.eq(target_label.cuda()).float()
            else:
                not_done =  adv_pred.eq(true_label.cuda()).float()
            success = (1 - not_done.detach().cpu()) * bool(float(np.sqrt(result.distance.value * image.size)) < args.epsilon)
            success_all.append(int(success[0].item()))
        result = {
                  'original_class': int(result.original_class),
                  'adversarial_class': int(result.adversarial_class),
                  'final_distance': np.sqrt(result.distance.value * image.size).item(),
                  'final_query_count': int(result._total_prediction_calls)}
        stats_distance = np.sqrt(attack.stats_distances * image.size)
        stats_query_count = attack.stats_query_counts
        for iteration, query_each_iteration in enumerate(stats_query_count):
            distortion_all[batch_index][int(query_each_iteration)] = stats_distance[iteration].item()
        result_json["statistical_details"][batch_index] = result
        # print progress
        # print_progress('Up to now:')

    result_json["distortion"] = distortion_all
    result_json["args"] = vars(args)
    result_json["success_all"] =  success_all,
    result_json["correct_all"] = correct_all
    # print finish information
    log.info('Attack finished.')
    with open(result_dump_path, "w") as result_file_obj:
        json.dump(result_json, result_file_obj, sort_keys=True)
    log.info("done, write stats info to {}".format(result_dump_path))

def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        # please note in this case, we need to import glog/logging after calling set_log_file(*, file_only=True)
        sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'boundary_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'boundary_attack-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname


# foolbox库在foolbox.zip里，调用foolbox库的代码在foolbox_attacks.py里

if __name__ == '__main__':
    # before going to the main function, we do following things:
    # 1. setup output directory
    # 2. make global variables: args, model (on cpu), loaders and device

    # 1. setup output directory
    args = parse_args()
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    # if args.num_part > 1, then this experiment is just a part and we should use the same token for all parts
    # to guarantee that, we use sha256sum of config in string format to generate unique token
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                              args.defense_norm,
                                                                              args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)  # # set log file, and import glog after that (since we might change sys.stdout/stderr on set_log_file())
    if args.attack_defense:
        assert args.defense_model is not None
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
        args.max_queries = 20000
    log.info('Foolbox package (version {}) imported from: {}'.format(foolbox.__version__, foolbox.__file__))
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args()

    # 2. make global variables
    # check device
    device = torch.device('cuda')
    args.device = str(device)
    # set random seed before init model
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.arch is not None
        archs = [args.arch]
    for arch in archs:
        args.arch = arch
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        log.info("After attack finished, the result json file will be dumped to {}".format(save_result_path))
        # do the business
        main(args, save_result_path)
