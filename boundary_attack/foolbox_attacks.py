#!/usr/bin/env python3
import sys
import os
import os.path as osp
import pickle
import argparse
import json
import random
from collections import OrderedDict
import string
import socket
import getpass
import copy
from datetime import datetime
import hashlib
import torch
import numpy as np

from models import make_victim_model
from loaders import make_loader

import biased_boundary_attack.foolbox as foolbox


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='output/debug', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist01', 'mnist', 'cifar10', 'imagenet'],
                        help='which dataset to use')
    parser.add_argument('--phase', default='test', type=str, choices=['train', 'val', 'valv2', 'test'],
                        help='train, val, test')
    parser.add_argument('--num-image', default=1000, type=int,
                        help='number of images to attack')
    parser.add_argument('--compare-with', default='', type=str,
                        help='specify reference experiment exp dir to make a fair comparison')
    parser.add_argument('--part-id', default=0, type=int,
                        help='args.part_id is the id of current part among all args.num_part')
    parser.add_argument('--num-part', default=1, type=int,
                        help='the task could be split in several parts, args.num_part is the total number of parts')
    parser.add_argument('--victim-arch', default='carlinet', type=str,
                        help='victim network architecture')
    parser.add_argument('--attack-type', default='untargeted', choices=['untargeted', 'targeted'],
                        help='type of attack, could be targeted or untargeted')
    parser.add_argument('--norm-type', default='l2', type=str, choices=['l2'],
                        help='l2 attack or linf attack')
    parser.add_argument('--attack-method', default='bapp', choices=['ba', 'cw', 'bapp'],
                        help='attack method')
    parser.add_argument('--save-all-steps', action='store_true',
                        help='save all intermediate adversarial images')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    # bapp (a.k.a., hsja) parameters
    parser.add_argument('--bapp-iteration', default=64, type=int,
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
    parser.add_argument('--ba-iteration', default=5000, type=int,
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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main():
    # make model
    log.info('Initializing model {} on {}'.format(args.victim_arch, args.dataset))
    model = make_victim_model(args.dataset, args.victim_arch, scratch=False).eval().to(device)
    if args.dataset == 'mnist01':
        num_classes = 2
    elif args.dataset in ['mnist', 'cifar10']:
        num_classes = 10
    elif args.dataset == 'imagenet':
        num_classes = 1000
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=num_classes, device=device)
    log.info('Foolbox model created')

    # make loader
    kwargs = dict()
    if args.dataset == 'imagenet':
        kwargs['size'] = model.input_size[-1]
    loader = make_loader(args.dataset, args.phase, 1, args.seed, **kwargs)  # batch size set to 1

    # extract image_id_ref from args.compare_with
    # args.compare_with should be the exp_dir of another experiment generated by policy_attack.py
    if len(args.compare_with) == 0:
        log.info('args.compare_with is not specified, so we do not load any initial point or image id')
        image_id_ref = None
    else:
        with open(osp.join(args.compare_with, 'config.json'), 'r') as f:
            compare_with_config = json.load(f)
        assert args.dataset == compare_with_config['dataset']
        assert args.phase == compare_with_config['phase']
        assert args.victim_arch == compare_with_config['victim_arch']
        image_id_ref = compare_with_config['image_id_ref']
        assert len(image_id_ref) > 0 and osp.exists(image_id_ref)
        log.info('We will load image ids from {}, which comes from config of args.compare_with ({})'.format(
            image_id_ref, args.compare_with))

    # load previously used image ids when training gradient model, if there are any
    used_image_ids = OrderedDict()
    used_image_ids['train_seen'] = list()
    used_image_ids['train_unseen'] = list()
    used_image_ids['test'] = list()
    if image_id_ref is not None:
        with open(osp.join(image_id_ref, 'config.json'), 'r') as f:
            image_id_ref_config = json.load(f)
        assert args.dataset == image_id_ref_config['dataset']
        assert args.phase == image_id_ref_config['phase']
        for key in used_image_ids.keys():
            fname = osp.join(image_id_ref, 'results', '{}_image_ids.pth'.format(key))
            used_image_ids[key] = torch.load(fname).tolist()
        log.info('Load used image ids from {}'.format(image_id_ref))
    for key, image_ids in used_image_ids.items():
        log.info('Found {} used image ids, key: {}'.format(len(image_ids), key))

    # these four variables represent type of visited images, and we treat them as boolean tensors
    # we use LongTensor instead of ByteTensor because we will do is_x.sum() later and ByteTensor will overflow
    is_correct = torch.LongTensor(0)
    is_ignore = torch.LongTensor(0)
    is_image_type = OrderedDict([
        ('train_seen', torch.LongTensor(0)),
        ('train_unseen', torch.LongTensor(0)),
        ('val', torch.LongTensor(0)),
        ('test', torch.LongTensor(0))
    ])
    num_image_each_type = OrderedDict([
        ('train_seen', min(args.num_image, len(used_image_ids['train_seen']))),
        ('train_unseen', 0),
        ('val', 0),
        ('test', args.num_image)
    ])
    log.info('Number of images included in the attacking task:')
    for key in num_image_each_type.keys():
        log.info('    {}: {}'.format(key, num_image_each_type[key]))

    # print function we call in the end of each iteration and when whole attack ends
    def print_progress(title):
        # print attack progress
        log.info(title)
        log.info('  visited images: {}'.format(is_correct.numel()))
        log.info('  ignored images: {}'.format(is_ignore.sum().item()))
        log.info('  correct images: {}'.format(is_correct.sum().item()))
        for key in is_image_type.keys():
            log.info('  correct {} images: {} not ignored, {} in total'.format(
                key, (is_correct & is_image_type[key] & (1 - is_ignore)).sum().item(),
                (is_correct & is_image_type[key]).sum().item()))

    # attack
    for batch_index, (image_id, image, label) in enumerate(loader):
        # batch attack is not supported yet
        assert image_id.numel() == 1

        # extract inputs
        assert image.ndimension() == 4
        assert image.shape[0] == 1
        image = image.numpy()[0]
        label = label.item()
        image_id = image_id.item()
        pred = int(np.argmax(fmodel.forward_one(image)))

        # for debug
        # if image_id != 8477:
        #     continue

        # append 0, and we will modify them later
        is_correct = torch.cat((is_correct, torch.LongTensor([0])))
        is_ignore = torch.cat((is_ignore, torch.LongTensor([0])))
        for image_type in is_image_type.keys():
            is_image_type[image_type] = torch.cat((is_image_type[image_type], torch.LongTensor([0])))

        # determine type of this image, use used_image_ids to determine the type
        if image_id in used_image_ids['train_seen']:
            # this image was used as training set in train_grad_model.py
            image_type = 'train_seen'
        elif image_id in used_image_ids['train_unseen']:
            # this image was not used as training set in train_grad_model.py
            # sometimes we also use these images to select the best model, so we can also treat them as 'val'
            # image_type = 'train_unseen'
            image_type = 'val'
        elif image_id in used_image_ids['test']:
            # this image was used to select the best model in train_grad_model.py
            image_type = 'val'
        else:
            # this image is brand new
            image_type = 'test'
        is_image_type[image_type][-1] = 1

        # fill in is_correct, we will use is_correct to check is_ignore later
        is_correct[-1] = pred == label

        # check whether we have visit enough images
        # we do not check this in the beginning of loop since values of is_* variables are not determined
        # if we have visited num_train_seen_image train images then we should ignore train_seen images later
        # if we have visited num_train_unseen_image train images then we should ignore train_unseen images later
        # if we have visited num_val_image val images then we should ignore val images later
        # if we have visited num_test_image test images then we should ignore test images later
        # then we should check args.num_part and args.part_id, if pass, we should go on to attack this image

        is_meet_each_type = OrderedDict([
            (key, (is_correct & is_image_type[key])[:-1].sum().item() >= num_image_each_type[key])
            for key in is_image_type.keys()
        ])
        if all(is_meet_each_type.values()):
            # attack task is done
            is_ignore[-1] = 1
            log.info('We have visited enough train_seen/train_unseen/val/test images, attack task is done')
            break
        if not is_correct[-1]:
            # if misclassified, we ignore directly
            is_ignore[-1] = 1
            log.info('Ignore {}-th image: image_id: {}, since it is misclassified'.format(
                batch_index, image_id))
        else:
            # current image is correctly classified
            assert is_image_type[image_type][-1].item()
            if is_meet_each_type[image_type]:
                # we've seen enough that type of images, so we should ignore
                is_ignore[-1] = 1
                log.info('Ignore {}-th image: image_id: {}, since we have visited enough ({}) {} images'.format(
                    batch_index, image_id, num_image_each_type[image_type], image_type))
            else:
                # we've not visited enough that type of images, now we should check part related arguments
                if args.num_part == 1:
                    # if there is only 1 part, we should go on to attack it
                    pass
                else:
                    # goes here indicates is_meet_each_type[key] is False, so there must be at least 1 image
                    assert num_image_each_type[image_type] > 0
                    num_image_each_part = num_image_each_type[image_type] // args.num_part
                    assert num_image_each_part > 0
                    current_num = (is_correct & is_image_type[image_type]).sum().item() - 1  # make index starts from 0
                    current_image_part_id = min(current_num // num_image_each_part, args.num_part - 1)
                    if current_image_part_id == args.part_id:
                        # current part matches args.part_id, we should run
                        pass
                    else:
                        is_ignore[-1] = 1
                        log.info(
                            'Ignore {}-th image: image_id: {}, since part of current image is {} '
                            '({} images for each part) while args.part_id is {}'.format(
                                batch_index, image_id, current_image_part_id,
                                num_image_each_part, args.part_id))

        # ignore image
        if is_ignore[-1].item():
            continue

        # start attack
        log.info('Attacking {}-th image: image_id: {}, image_type: {}'.format(batch_index, image_id, image_type))

        # find initial point using blended uniform noise
        # set random seed based on image_id, so we will have the same starting point for different attacking algorithms
        random.seed(image_id)
        np.random.seed(image_id)
        torch.manual_seed(image_id)
        torch.cuda.manual_seed(image_id)
        torch.cuda.manual_seed_all(image_id)

        # load init point and init query
        init_adv_image = init_distance = None
        init_query_count = 0
        if len(args.compare_with) > 0:
            fname = osp.join(args.compare_with, 'results', 'image-id-{}.pkl'.format(image_id))
            assert osp.exists(fname)
            with open(fname, 'rb') as f:
                prev_result = pickle.load(f)
                init_adv_image = prev_result['init_adv_image']
                assert init_adv_image.ndimension() == 4
                assert init_adv_image.shape[0] == 1
                init_adv_image = init_adv_image.numpy()[0]
                init_distance = prev_result['init_distance']
                init_query_count = prev_result['init_query_count']
            log.info('{}-th image: image_id {}, use initial point from {}, dist: {:.4f}'.format(
                batch_index, image_id, fname, init_distance))
        else:
            log.info('{}-th image: image_id {},'.format(batch_index, image_id) +
                     ' args.compare_with is not specified, so we will use default foolbox initialization')

        # initialize attack object and perform attack
        if args.attack_type == 'untargeted':
            criterion = foolbox.criteria.Misclassification()
        else:
            criterion = foolbox.criteria.TargetClass((label + 1) % loader.num_class)

        if args.attack_method == 'ba':
            attack = foolbox.attacks.BoundaryAttack(fmodel, criterion=criterion)
            with torch.no_grad():
                result = attack(input_or_adv=image,
                                label=label,
                                unpack=False,
                                iterations=args.ba_iteration,
                                max_directions=args.ba_max_directions,
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
        assert result.distance.value > 0, 'Failed to attack image_id: {}'.format(image_id)
        log.info('Attack {}-th image done, image_id {}, image_type {}'.format(batch_index, image_id, image_type))
        log.info('   init query count: {}'.format(init_query_count))
        if init_distance is not None:
            log.info('      init distance: {:.4f}'.format(init_distance))
        else:
            log.info('      init distance: None')
        log.info('  final query count: {}'.format(result._total_prediction_calls))
        log.info('     final distance: {:.4g} ({})'.format(result.distance.value, result.distance.name()))
        log.info('     final distance: {:.4f}'.format(np.sqrt(result.distance.value * image.size)))
        log.info('              label: {}'.format(label))
        log.info('               pred: {}'.format(pred))
        log.info('          adv label: {}'.format(result.adversarial_class))

        # save results
        log.info('Final result for {}-th image: image_id: {}, query: {:d}, dist: {:.4f}'.format(
            batch_index, image_id, result._total_prediction_calls, np.sqrt(result.distance.value * image.size)))
        fname = osp.join(args.exp_dir, 'results', 'image-id-{}.pkl'.format(image_id))
        os.makedirs(osp.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            result = {'unperturbed': result.unperturbed,
                      'perturbed': result.perturbed,
                      'original_class': result.original_class,
                      'adversarial_class': result.adversarial_class,
                      'final_distance': np.sqrt(result.distance.value * image.size),
                      'final_query_count': result._total_prediction_calls,
                      'image_type': image_type}
            if args.attack_method in ['ba', 'bapp']:
                result.update({'distance': np.sqrt(attack.stats_distances * image.size),
                               'query_count': attack.stats_query_counts + init_query_count,
                               'init_adv_image': init_adv_image,
                               'init_distance': init_distance,
                               'init_query_count': init_query_count})
            if args.save_all_steps:
                result.update({'all_steps': attack.stats_all_steps})
            json.dump(result, f)
        log.info('Result for current image saved to {}'.format(fname))

        # print progress
        print_progress('Up to now:')

    # finished, create empty file thus others could check whether or not this task is done
    open(osp.join(args.exp_dir, 'done'), 'a').close()

    # print finish information
    log.info('Attack finished.')


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


def get_random_dir_name(seed=None):
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice(vocab) for _ in range(8))


# foolbox库在foolbox.zip里，调用foolbox库的代码在foolbox_attacks.py里

if __name__ == '__main__':
    # before going to the main function, we do following things:
    # 1. setup output directory
    # 2. make global variables: args, model (on cpu), loaders and device

    # 1. setup output directory
    args = parse_args()

    # if args.num_part > 1, then this experiment is just a part and we should use the same token for all parts
    # to guarantee that, we use sha256sum of config in string format to generate unique token
    assert 0 <= args.part_id < args.num_part <= args.num_image
    token = copy.deepcopy(vars(args))
    del token['part_id']
    del token['exp_dir']
    token = json.dumps(token, sort_keys=True, indent=4)
    token = hashlib.sha256(token.encode('utf-8')).digest()  # type(token) == bytes
    token = get_random_dir_name(seed=token)
    token = '-'.join([datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                      token,
                      'part-{}-in-{}'.format(args.part_id, args.num_part)])
    args.exp_dir = osp.join(args.exp_dir, token)
    os.makedirs(args.exp_dir, exist_ok=True)

    # set log file, and import glog after that (since we might change sys.stdout/stderr on set_log_file())
    set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=args.ssh)
    import glog as log
    log.info('Foolbox package (version {}) imported from: {}'.format(foolbox.__version__, foolbox.__file__))
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.exp_dir))
    os.system('cp -r *.py models {}'.format(args.exp_dir))

    # 2. make global variables

    # check device
    device = torch.device('cuda')

    # set random seed before init model
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # do the business
    main()
