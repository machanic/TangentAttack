import sys
import os
import os.path as osp
import socket
import functools
import getpass
import argparse
import json
import random
import copy
from collections import OrderedDict
from glob import glob
import glog as log
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from models import standard_model, defensive_model
from dataset.dataset_loader_maker import DataLoaderMaker
# record best model
from models.standard_model import StandardModel
from policy_driven_attack.policy_model import make_policy_model
from policy_driven_attack.pre_training.grad_dataset import GradDataset

best_train_unseen_tuple_sim = 0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='output/debug', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--grad-dir', default='', type=str,
                        help='directory where gradients are stored')
    parser.add_argument('--num-train', default=1000, type=int,
                        help='number of image ids in training')
    parser.add_argument('--max-query', default=20000, type=int,
                        help='maximum number of queries allowed')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='save model every epoch')
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist01', 'mnist', 'cifar10', 'imagenet'],
                        help='which dataset to use')
    parser.add_argument('--phase', default='test', type=str, choices=['train', 'val', 'valv2', 'test'],
                        help='train, val, test')
    parser.add_argument('--num-test', default=100, type=int,
                        help='number of test images')
    parser.add_argument('--victim-arch', default='carlinet', type=str,
                        help='victim network architecture')
    parser.add_argument('--policy-arch', default='empty', type=str,
                        help='policy network architecture')
    parser.add_argument('--policy-weight-fname', default='', type=str,
                        help='pre-trained policy weight filename')
    parser.add_argument('--policy-init-std', default=3e-3, type=float,
                        help='initial value of std for policy network')
    parser.add_argument('--policy-bilinear', action='store_true',
                        help='use bilinear in policy network if applicable')
    parser.add_argument('--policy-normalization-type', default='none', type=str, choices=['none', 'bn', 'gn'],
                        help='normalization type in policy network if applicable')
    parser.add_argument('--policy-use-tanh', action='store_true',
                        help='use tanh in policy network if applicable')
    parser.add_argument('--policy-base-width', default=16, type=int,
                        help='set base width parameter in policy network if applicable')
    parser.add_argument('--policy-calibrate', action='store_true',
                        help='calibrate output of policy network using mean in policy network if applicable')
    parser.add_argument('--grad-size', default=0, type=int,
                        help='force to use a specific shape for grad')
    parser.add_argument('--use-true-grad', action='store_true',
                        help='use true gradient instead of estimated gradient for training')
    parser.add_argument('--image-gaussian-std', default=0.0, type=float,
                        help='add gaussian noise for data augmentation')
    parser.add_argument('--grad-gaussian-std', default=0.0, type=float,
                        help='add gaussian noise for data augmentation')
    parser.add_argument('--no-init-eval', action='store_true',
                        help='do not evaluate performance after model initialization')
    parser.add_argument('--ce-lmbd', nargs=4, default=[1.0, 0.01, 0.01, 0.01],
                        help='cross entropy coefficient in loss, only applied if policy net could output logit')
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'],
                        help='type of optimizer')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--step-mult', nargs='+', default=[0.1, 0.01],
                        help='multiplier for step lr policy')
    parser.add_argument('--step-at', nargs='+', default=[250, 350],
                        help='step at specified epochs')
    parser.add_argument('--decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--clip-grad', default=5., type=float,
                        help='max gradient norm')
    parser.add_argument('--batch-size', default=50, type=int,
                        help='batch size during training and testing')
    parser.add_argument('--print-freq', default=1, type=int,
                        help='print each args.print_freq batches')
    parser.add_argument('--num-worker', default=4, type=int,
                        help='number of workers used for gradient data loader')
    parser.add_argument('--pre-load', action='store_true',
                        help='pre-load all grad training data into cpu memory before training')
    parser.add_argument('--check-unseen-tuple', action='store_true',
                        help='if set to True, we will also load and check train_unseen tuple sim')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh. '
                             'If set to True, we will not print anything to screen and only redirect them to log file')

    # data parallel parameters
    parser.add_argument('--use-ddp', action='store_true',
                        help='Use pytorch ddp')
    parser.add_argument('--ddp-world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--ddp-rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--ddp-url', default='tcp://127.0.0.1:12345', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--ddp-backend', default='nccl', type=str,
                        help='distributed backend')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.step_at = list(map(int, args.step_at))
    args.step_mult = list(map(float, args.step_mult))
    args.ce_lmbd = list(map(float, args.ce_lmbd))
    return args


def norm(v, p=2):
    v = v.view(v.shape[0], -1)
    if p == 2:
        return torch.clamp(v.norm(dim=1).view(-1, 1, 1, 1), min=1e-8)
    elif p == 1:
        return torch.clamp(v.abs().sum(dim=1).view(-1, 1, 1, 1), min=1e-8)
    else:
        raise ValueError('Unknown norm p={}'.format(p))

def test_tuple(train_phases, grad_loader, model, policy, gpu, args):
    # check sim of predicted gradients on saved grad tuples
    log.info('Testing sim on tuples @ gpu {}, phase: {}'.format(gpu, train_phases))
    policy.eval()
    result = dict()
    with torch.no_grad():
        for phase in train_phases:
            num_tuple = len(grad_loader[phase].dataset)
            log.info('Testing sim on {} tuples @ gpu {}, {} in total'.format(phase, gpu, num_tuple))
            # if args.dataset == 'imagenet' and phase == 'train_seen':
            #     log.info('Ignore train_seen for imagenet')
            #     result['train_seen_tuple_sim'.format(phase)] = 0
            #     continue
            sim_all = torch.zeros(num_tuple)

            for batch_index, (adv_image, image, label, grad) in enumerate(grad_loader[phase]):
                # move inputs to device
                adv_image = adv_image.cuda(gpu)
                image = image.cuda(gpu)
                label = label.cuda(gpu)
                grad = grad.cuda(gpu)

                # get target label
                logit = model(adv_image)
                argsort = logit.argsort(dim=1, descending=True)
                target = argsort[:, 0] * (1 - argsort[:, 0].eq(label)).long() +\
                         argsort[:, 1] * (argsort[:, 0].eq(label)).long()
                assert not label.eq(target).any().item()

                # make select indicator
                selected = torch.arange(batch_index * grad_loader[phase].batch_size,
                                        batch_index * grad_loader[phase].batch_size + adv_image.shape[0])

                # get estimated gradients
                pred = policy(adv_image, image, label, target, output_fields=('grad',))['grad']

                if pred.shape != grad.shape:
                    pred = F.interpolate(pred, size=grad.shape[-1], mode='bilinear', align_corners=True)
                assert grad.shape == pred.shape
                sim = (grad * pred).sum(dim=(1, 2, 3)) / norm(grad).view(-1) / norm(pred).view(-1)
                sim_all[selected] = sim.detach().cpu()

                n = batch_index * args.batch_size + adv_image.shape[0]
                if batch_index % args.print_freq == 0 or batch_index == len(grad_loader[phase]) - 1:
                    log.info('Performance of batch {} @ gpu {} (tuple {} - {}):'.format(
                        batch_index, gpu, n - adv_image.shape[0], n))
                    log.info('    {}: {} / {} @ gpu {}, tuple sim: {:.4f}'.format(
                        phase, n, num_tuple, gpu, sim_all[:n].mean()))

                    # clear gpu memory cache
                    if args.dataset == 'imagenet':
                        torch.cuda.empty_cache()

                if args.dataset == 'imagenet' and n >= 10000:
                    log.info('Early break for imagenet')
                    break

            # save result for current phase
            result['{}_tuple_sim'.format(phase)] = sim_all.mean().item()

    log.info('Performance of current model (tuple sim) @ gpu {}:'.format(gpu))
    max_len = max(list(map(lambda t: len(t), train_phases)))
    for phase in train_phases:
        log.info(' ' * (max_len + 1 - len(phase)) + ' {} tuple sim @ gpu {}: {:.4f}'.format(
            phase.replace('_', ' '), gpu, result['{}_tuple_sim'.format(phase)]))

    return result


def test(train_image_ids, stage1_image_ids, loader, model, policy, gpu, num_gpu_per_node, args):
    # check sim of predicted gradients on clean images
    log.info('Testing sim on clean images')
    policy.eval()
    sim_train_seen, train_seen_index = torch.zeros(len(train_image_ids['train_seen'])), 0
    sim_train_unseen, train_unseen_index = torch.zeros(len(train_image_ids['train_unseen'])), 0
    sim_test, test_index = torch.zeros(args.num_test), 0
    assert sim_train_seen.shape[0] + sim_train_unseen.shape[0] == len(stage1_image_ids)

    # record test image ids to make sure we use the same test set across each epoch
    # note we do not guarantee the test set keeps the same across different experiments, since the number of stage1
    # images might be different across different experiments
    test_image_ids = list()

    # start test
    with torch.no_grad():
        for batch_index, (image_id, image, label) in enumerate(loader):
            # move inputs to device
            image = image.cuda(gpu)
            label = label.cuda(gpu)

            # temporarily allows gradient calculation to get logit and true grad
            with torch.enable_grad():
                image.requires_grad = True
                logit = model(image)
                acc = logit.argmax(dim=1).eq(label)

                # get true gradients using logit diff (cw) loss
                # we only use correctly classified images, others will not be counted
                target = logit.argsort(dim=1, descending=True)[:, 1]
                loss = logit[torch.arange(image.shape[0]), target] - logit[torch.arange(image.shape[0]), label]
                true_grad = torch.autograd.grad(loss.mean() * image.shape[0], image)[0]
                true_grad = true_grad.detach()
                image.requires_grad = False

            # get estimated gradients
            pred = policy(image, image, label, target, output_fields=('grad',))['grad']
            if pred.shape != true_grad.shape:
                pred = F.interpolate(pred, size=true_grad.shape[-1], mode='bilinear', align_corners=True)

            # calculate cosine similarity
            assert true_grad.shape == pred.shape
            sim = (true_grad * pred).sum(dim=(1, 2, 3)).abs() / norm(true_grad).view(-1) / norm(pred).view(-1)

            # assign each similarity into train_seen, train_unseen, test
            for image_index_in_batch in range(image_id.shape[0]):
                t = int(image_id[image_index_in_batch].item())
                if t in train_image_ids['train_seen']:
                    if train_seen_index < sim_train_seen.shape[0]:
                        sim_train_seen[train_seen_index] = sim[image_index_in_batch].cpu()
                        train_seen_index += 1
                elif t in train_image_ids['train_unseen']:
                    if train_unseen_index < sim_train_unseen.shape[0]:
                        sim_train_unseen[train_unseen_index] = sim[image_index_in_batch].cpu()
                        train_unseen_index += 1
                else:
                    if test_index < sim_test.shape[0] and acc[image_index_in_batch].item():
                        sim_test[test_index] = sim[image_index_in_batch].cpu()
                        test_index += 1
                        test_image_ids.append(t)

            n = batch_index * args.batch_size + image.shape[0]
            if batch_index % args.print_freq == 0 or test_index >= args.num_test:
                log.info('Performance of batch {} @ gpu {} (image {} - {}):'.format(
                    batch_index, gpu, n - image.shape[0], n))
                log.info('    train seen @ gpu {}: {} / {}, sim: {:.4f}'.format(
                    gpu, train_seen_index, sim_train_seen.shape[0], sim_train_seen[:train_seen_index].mean().item()))
                log.info('  train unseen @ gpu {}: {} / {}, sim: {:.4f}'.format(
                    gpu, train_unseen_index, sim_train_unseen.shape[0],
                    sim_train_unseen[:train_unseen_index].mean().item()))
                log.info('     test seen @ gpu {}: {} / {}, sim: {:.4f}'.format(
                    gpu, test_index, sim_test.shape[0], sim_test[:test_index].mean().item()))

                # clear gpu memory cache
                if args.dataset == 'imagenet':
                    del image, label, logit, acc, target, loss, true_grad, pred, sim
                    torch.cuda.empty_cache()

            if args.dataset == 'imagenet' and n >= 10000:
                log.info('Early break for imagenet')
                break

            # check test done
            if test_index >= args.num_test:
                assert train_seen_index >= sim_train_seen.shape[0]
                assert train_unseen_index >= sim_train_unseen.shape[0]
                log.info('We have tested {} images @ gpu {}, break'.format(args.num_test, gpu))
                break

    log.info('Performance of current model (clean sim) @ gpu {}:'.format(gpu))
    log.info('    train seen sim @ gpu {}: {:.4f}'.format(gpu, sim_train_seen.mean().item()))
    log.info('  train unseen sim @ gpu {}: {:.4f}'.format(gpu , sim_train_unseen.mean().item()))
    log.info('          test sim @ gpu {}: {:.4f}'.format(gpu, sim_test.mean().item()))

    # check test image ids on rank 0 node
    if (not args.use_ddp) or (args.use_ddp and args.ddp_rank % num_gpu_per_node == 0):
        test_image_ids = torch.LongTensor(list(set(test_image_ids)))
        if args.dataset != 'imagenet':
            # we might early break for imagenet, so the number of test image ids might not be accurate
            assert test_image_ids.shape[0] == args.num_test
        test_image_ids, _ = test_image_ids.sort()
        fname = osp.join(args.exp_dir, 'results', 'test_image_ids.pth')
        if osp.exists(fname):
            prev_test_image_ids = torch.load(fname)
            assert (prev_test_image_ids == test_image_ids).all().item()
        else:
            os.makedirs(osp.dirname(fname), exist_ok=True)
            torch.save(test_image_ids, fname)

    # return avg results
    return {'train_seen_sim': sim_train_seen.mean().item(),
            'train_unseen_sim': sim_train_unseen.mean().item(),
            'test_sim': sim_test.mean().item()}


def train(optimizer, epoch_index, train_loader, model, policy, gpu, args):
    # set model's training flag
    policy.train()

    # initialize stat variables
    if args.use_ddp:
        # in this case, DistributedSampler is used, so we only visit part of the whole dataset
        num_tuple = train_loader.sampler.num_samples
    else:
        # we will visit the whole dataset
        num_tuple = len(train_loader.dataset)
    sim_all = torch.zeros(num_tuple)
    grad_all = torch.zeros(num_tuple)
    lce_all = torch.zeros(num_tuple)
    tce_all = torch.zeros(num_tuple)
    ce_all = torch.zeros(num_tuple)
    loss_all = torch.zeros(num_tuple)
    clip_grad_all = torch.zeros(num_tuple)

    # output these fields from policy network
    is_warmup = epoch_index < args.warmup_epochs
    if args.policy_arch.endswith('_inv'):
        if is_warmup:
            output_fields = ('adv_logit', 'logit')
        else:
            output_fields = ('grad', 'adv_logit', 'logit')
    else:
        assert not is_warmup
        output_fields = ('grad',)

    # start train
    for batch_index, (adv_image, image, label, grad) in enumerate(train_loader):
        # move training data into device
        adv_image = adv_image.cuda(gpu)
        image = image.cuda(gpu)
        label = label.cuda(gpu)
        grad = grad.cuda(gpu)

        # for debug: generate true grad online
        # adv_image.requires_grad = True
        # logit = model(adv_image)
        # target = logit.argsort(dim=1, descending=True)[:, 0]
        # loss = logit[torch.arange(adv_image.shape[0]), target] - logit[torch.arange(adv_image.shape[0]), label]
        # true_grad = torch.autograd.grad(loss.mean() * adv_image.shape[0], adv_image)[0]
        # true_grad = true_grad.detach()
        # adv_image.requires_grad = False

        # get target label
        # some logit values are very close, and thus target = logit.argmax(dim=1) is not numerically stable
        logit = model(adv_image)
        argsort = logit.argsort(dim=1, descending=True)
        target = argsort[:, 0] * (1 - argsort[:, 0].eq(label)).long() + \
                 argsort[:, 1] * (argsort[:, 0].eq(label)).long()
        assert not label.eq(target).any().item()

        # perform gaussian data augmentation after calculating target
        if args.image_gaussian_std > 0:
            noise = torch.randn(*adv_image.shape).to(logit.device)
            noise *= adv_image.abs().mean(dim=(1, 2, 3)).view(adv_image.shape[0], 1, 1, 1)
            noise *= args.image_gaussian_std
            adv_image += noise
            adv_image = torch.clamp(adv_image, 0, 1)

        if args.grad_gaussian_std > 0:
            noise = torch.randn(*grad.shape).to(logit.device)
            noise *= grad.abs().mean(dim=(1, 2, 3)).view(grad.shape[0], 1, 1, 1)
            noise *= args.grad_gaussian_std
            grad += noise

        assert 0 < adv_image.shape[0] == image.shape[0] == label.shape[0] == \
               grad.shape[0] <= train_loader.batch_size == args.batch_size

        # set selected variable, we will use this to compute epoch_mean
        selected = torch.arange(batch_index * train_loader.batch_size,
                                batch_index * train_loader.batch_size + image.shape[0])

        # calculate loss and do backward
        output = policy(adv_image, image, label, target, output_fields=output_fields)

        if 'grad' in output:
            pred = output['grad']
            if pred.shape != grad.shape:
                pred = F.interpolate(pred, size=grad.shape[-1], mode='bilinear', align_corners=True)
            assert grad.shape == pred.shape
            sim = (grad * pred).sum(dim=(1, 2, 3)) / norm(grad).view(-1) / norm(pred).view(-1)
            sim_all[selected] = sim.detach().cpu()

        # combine and make loss
        if args.policy_arch.endswith('_inv'):
            # for inv networks, we also need to calculate ce / lce / tce
            if is_warmup:
                # current epoch is a warmup epoch, so we should discard sim in loss
                ce_lmbd = [0.0, float(args.ce_lmbd[1] > 0), 0.5, 0.5]
            else:
                # current epoch is a normal epoch, so we need to count in sim in loss
                ce_lmbd = args.ce_lmbd

            # logit for adv_image
            adv_logit = output['adv_logit']
            lce = F.cross_entropy(adv_logit, label, reduction='none')
            tce = F.cross_entropy(adv_logit, target, reduction='none')
            lce_all[selected] = lce.detach().cpu()
            tce_all[selected] = tce.detach().cpu()

            # logit for clean_image
            logit = output['logit']
            ce = F.cross_entropy(logit, label, reduction='none')
            ce_all[selected] = ce.detach().cpu()

            # combine and make loss
            if is_warmup:
                loss = ce * ce_lmbd[1] + lce * ce_lmbd[2] + tce * ce_lmbd[3]
            else:
                assert 'grad' in output
                loss = -sim * ce_lmbd[0] + ce * ce_lmbd[1] + lce * ce_lmbd[2] + tce * ce_lmbd[3]
        else:
            # policy network will not output logit and adv_logit, so we can only penalize sim
            assert not is_warmup
            loss = -sim
        loss_all[selected] = loss.detach().cpu()

        optimizer.zero_grad()
        policy.zero_grad()
        loss.mean().backward()

        # clip grad norm
        grad_all[selected] = torch.nn.utils.clip_grad_norm_(
            functools.reduce(lambda g1, g2: g1['params'] + g2['params'], optimizer.param_groups),
            args.clip_grad, norm_type=2)
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    clip_grad_all[selected] += p.grad.detach().norm(2) ** 2
        clip_grad_all[selected] = torch.sqrt(clip_grad_all[selected])

        # update weights of policy network
        optimizer.step()

        # log
        if batch_index < args.print_freq or \
                batch_index % args.print_freq == 0 or \
                batch_index == len(train_loader) - 1:
            log.info('Processing batch {} @ gpu {}, tuple {} - {} / {}'.format(
                batch_index, gpu, selected[0].item(), selected[0].item() + image.shape[0], num_tuple))
            if args.policy_arch.endswith('_inv'):
                keys = ['sim', 'lce', 'tce', 'ce',  'loss', 'grad', 'clip_grad']
            else:
                keys = ['sim', 'loss', 'grad', 'clip_grad']
            max_len = max(list(map(lambda t: len(t), keys)))
            for key in keys:
                value = eval('{}_all'.format(key))
                epoch_mean = (value.sum() / (selected[-1] + 1)).item()
                batch_mean = value[selected].mean().item()
                log.info(' ' * (max_len + 1 - len(key)) + key +
                         ' @ gpu {}: epoch {:.4f}, batch {:.4f}'.format(gpu, epoch_mean, batch_mean))

def make_model_parallel(name, model, use_ddp, gpu):
    assert gpu is not None, 'GPU id must be specified'
    if use_ddp:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        log.info('Deploy {} model on gpu {} (ddp is used)'.format(name, gpu))
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
        log.info('Deploy {} model on gpu {} (ddp is not used)'.format(name, gpu))
    return model

def main_worker(gpu, num_gpu_per_node, args):
    global best_train_unseen_tuple_sim

    # gpu for this worker
    log.info('Use GPU: {} for training'.format(gpu))

    # init process group
    if args.use_ddp:
        # adjust local rank (on current machine) to global rank (among all machines)
        assert args.ddp_rank >= 0
        global_ddp_rank = args.ddp_rank * num_gpu_per_node + gpu
        log.info('Set args.ddp_rank from local rank {} to global rank {}, since ddp is used'.format(
            args.ddp_rank, global_ddp_rank))
        args.ddp_rank = global_ddp_rank
        del global_ddp_rank

        # initialize ddp, connect to other processes
        dist.init_process_group(backend=args.ddp_backend, init_method=args.ddp_url,
                                world_size=args.ddp_world_size, rank=args.ddp_rank)

    # make victim model
    model = StandardModel(args.dataset, args.victim_arch, no_grad=False, load_pretrained=True).eval()
    input_size = model.input_size
    model = make_model_parallel('victim', model, args.use_ddp, gpu)

    # make policy model
    kwargs = OrderedDict({'init_std': args.policy_init_std, 'input_size': args.grad_size})
    if len(args.policy_weight_fname) > 0:
        if args.policy_arch == 'empty':
            log.info('Ignore args.policy_weight_fname: {}, since policy arch is empty'.format(args.policy_weight_fname))
        else:
            kwargs['weight_fname'] = args.policy_weight_fname
    if args.policy_arch in ['unet']:
        kwargs['bilinear'] = args.policy_bilinear
        kwargs['normalization_type'] = args.policy_normalization_type
        kwargs['base_width'] = args.policy_base_width
    elif args.policy_arch.endswith('_inv'):
        kwargs['calibrate'] = args.policy_calibrate
        kwargs['use_tanh'] = args.policy_use_tanh
        if args.policy_arch.startswith('vgg') or args.policy_arch.startswith('resnet'):
            kwargs['normalization_type'] = args.policy_normalization_type

    policy_keys = OrderedDict()
    for key, value in vars(args).items():
        if key.startswith('policy_') and key != 'policy_arch':
            policy_keys[key] = value
    log.info('Found policy keys: {}, only use {} for dataset {} and arch {}'.format(
        policy_keys, kwargs, args.dataset, args.policy_arch))
    policy = make_policy_model(args.dataset, args.policy_arch, **kwargs).train()
    log.info('Policy network:')
    log.info(policy)
    policy = make_model_parallel('policy', policy, args.use_ddp, gpu)

    # adjust batch size and number of workers in data loader since we might have more than 1 gpu if ddp is used
    if args.use_ddp:
        log.info('Adjust batch size and number of workers since ddp is used')
        batch_size = int(args.batch_size / num_gpu_per_node)
        num_worker = int((args.num_worker + num_gpu_per_node - 1) / num_gpu_per_node)
        log.info('Set batch_size from {} to {}'.format(args.batch_size, batch_size))
        log.info('Set workers from {} to {}'.format(args.num_worker, num_worker))
        args.batch_size = batch_size
        args.num_worker = num_worker
        del batch_size, num_worker

    # make loader
    kwargs = dict()
    if args.dataset == 'imagenet':
        kwargs['size'] = input_size[-1]
    loader = DataLoaderMaker.get_imgid_img_label_data_loader(args.dataset, args.batch_size,args.phase=="train",shuffle=args.phase=='train')

    # load image_ids from args.grad_dir
    log.info('Loading stage1 image ids from {}'.format(args.grad_dir))
    pattern = osp.join(args.grad_dir, 'results', 'saved_grads', 'image-id-*')
    stage1_image_ids = glob(pattern)
    stage1_image_ids = list(map(lambda t: int(t.split('/')[-1].split('-')[-1]), stage1_image_ids))
    stage1_image_ids = list(set(stage1_image_ids))
    assert len(stage1_image_ids) >= args.num_train, 'Require {} image ids, but only found {}, exit'.format(
        args.num_train, len(stage1_image_ids))
    stage1_image_ids.sort()  # make sure we always perform the same split on stage1_image_ids
    if (not args.use_ddp) or (args.use_ddp and args.ddp_rank % num_gpu_per_node == 0):
        fname = osp.join(args.exp_dir, 'results', 'stage1_image_ids.pth')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        torch.save(torch.LongTensor(stage1_image_ids), fname)
    log.info('Found {} image ids in stage1 using pattern {}'.format(len(stage1_image_ids), pattern))

    # random split stage1 image ids into train_seen and train_unseen, and train_seen should have args.num_train images
    # we do the split regardless of args.check_unseen_tuple
    log.info('Splitting {} stage1 image ids into train_seen and train_unseen'.format(len(stage1_image_ids)))
    state = np.random.get_state()
    np.random.seed(0)
    perm = np.random.permutation(len(stage1_image_ids))
    np.random.set_state(state)
    train_image_ids = dict()
    for phase in ['train_seen', 'train_unseen']:
        if phase == 'train_seen':
            train_image_ids[phase] = np.array(stage1_image_ids)[perm[:args.num_train]].tolist()
        else:
            assert phase == 'train_unseen'
            train_image_ids[phase] = np.array(stage1_image_ids)[perm[args.num_train:]].tolist()
        train_image_ids[phase].sort()
        log.info('Sample {} {} image ids from {}'.format(len(train_image_ids[phase]), phase, len(stage1_image_ids)))

        # save image ids to disk
        if (not args.use_ddp) or (args.use_ddp and args.ddp_rank % num_gpu_per_node == 0):
            fname = osp.join(args.exp_dir, 'results', '{}_image_ids.pth'.format(phase))
            os.makedirs(osp.dirname(fname), exist_ok=True)
            torch.save(torch.LongTensor(train_image_ids[phase]), fname)

    # load (adv_image, grad) tuple filenames for selected image_ids
    if args.check_unseen_tuple:
        train_phases = ['train_seen', 'train_unseen']
    else:
        train_phases = ['train_seen']
    log.info('Loading grad tuples for {}'.format(' and '.join(train_phases)))
    train_tuple_fnames = dict()
    for phase in train_phases:
        train_tuple_fnames[phase] = list()
        num_before_filtered = 0
        for train_image_id in train_image_ids[phase]:
            pattern = osp.join(args.grad_dir, 'results', 'saved_grads', 'image-id-{}/step-*.pth'.format(train_image_id))
            t = glob(pattern)
            num_before_filtered += len(t)

            # filter saved grads using args.max_query
            with open(osp.join(args.grad_dir, 'results', 'image-id-{}.pkl'.format(train_image_id)), 'rb') as f:
                max_step = (pickle.load(f)['query_count'] <= args.max_query).nonzero().max().item()
            t = list(filter(lambda tt: int(tt.split('/')[-1].split('.')[0].split('-')[-1]) <= max_step, t))

            t.sort()  # make sure the order of training data keeps the same across different runs
            train_tuple_fnames[phase] += t
        log.info('Found {} tuples ({} before filter) for {} ({} image ids), each image has {:.1f} tuples on avg'.format(
            len(train_tuple_fnames[phase]), num_before_filtered, phase, len(train_image_ids[phase]),
            len(train_tuple_fnames[phase]) / len(train_image_ids[phase])))

    # create loader for grad tuples
    log.info('Start to initialize dataset and data loader for {} (pre_load={})'.format(
        ' and '.join(train_phases), args.pre_load))
    grad_dataset = dict()
    grad_loader = dict()
    for phase in train_phases:
        grad_dataset[phase] = GradDataset(
            train_tuple_fnames[phase], args.use_true_grad if phase == 'train_seen' else True, args.pre_load)
        log.info('Grad dataset for {} initialization done, use_true_grad: {}'.format(
            phase, grad_dataset[phase].use_true_grad))

        # the grad loader is used for training, so we do not mind if the last few samples are missing
        # in some cases, the last batch contains only 1 image and may cause some peaks in the training process figure
        grad_loader[phase] = torch.utils.data.DataLoader(
            grad_dataset[phase], batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_worker, pin_memory=True, drop_last=False)
        log.info('Grad data loader for {} initialization done'.format(phase))

    if args.use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(grad_dataset['train_seen'])
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        grad_dataset['train_seen'], batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_worker, pin_memory=True, drop_last=False, sampler=train_sampler)

    # pretest
    if not args.no_init_eval:
        log.info('Evaluating performance before training')
        test_tuple(train_phases, grad_loader, model, policy, gpu, args)
        test(train_image_ids, stage1_image_ids, loader, model, policy, gpu, num_gpu_per_node, args)

    # make optimizer
    def trainable(name):
        if name.split('.')[-1] == 'normal_mean':
            return True
        elif name.split('.')[-1] in ['normal_logstd', 'empty_coeff', 'empty_normal_mean']:
            return False
        else:
            return True

    param_groups = list()
    param_groups.append({'params': [p[1] for p in policy.named_parameters() if trainable(p[0]) and 'bias' not in p[0]],
                         'lr': args.lr, 'weight_decay': args.decay})
    param_groups.append({'params': [p[1] for p in policy.named_parameters() if trainable(p[0]) and 'bias' in p[0]],
                         'lr': args.lr, 'weight_decay': 0.0})
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    optimizer_init_state_dict = copy.deepcopy(optimizer.state_dict())
    log.info('Optimizer: {}'.format(optimizer))
    log.info('Optimizer init state: {}'.format(optimizer_init_state_dict))
    log.info('Number of params to be optimized: {}'.format(
        [len(param_group['params']) for param_group in param_groups]))

    log.info('Training policy network to simulate gradient')
    for epoch_index in range(args.epochs):
        if args.use_ddp:
            train_sampler.set_epoch(epoch_index)

        if epoch_index < args.warmup_epochs:
            log.info('Training for {}-th epoch (warmup epoch)'.format(epoch_index))
        else:
            log.info('Training for {}-th epoch (normal epoch)'.format(epoch_index))

        # reset optimizer if warmup ends
        if epoch_index == args.warmup_epochs > 0:
            log.info('Reset optimizer because warmup ends')
            optimizer.load_state_dict(optimizer_init_state_dict)
            log.info('Optimizer state: {}'.format(optimizer.state_dict()))

        # adjust learning rate
        if epoch_index in args.step_at and args.optimizer in ['SGD']:
            # Adam does not change learning rate
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                new_lr = args.lr * args.step_mult[args.step_at.index(epoch_index)]
                param_group['lr'] = new_lr
                log.info('Epoch {}, cut learning rate from {:g} to {:g}'.format(epoch_index, lr, new_lr))

        train(optimizer, epoch_index, train_loader, model, policy, gpu, args)

        # evaluate and save current model
        log.info('Evaluating epoch {}'.format(epoch_index))
        result = test_tuple(train_phases, grad_loader, model, policy, gpu, args)
        test(train_image_ids, stage1_image_ids, loader, model, policy, gpu, num_gpu_per_node, args)
        if result.get('train_unseen_tuple_sim', -1) > best_train_unseen_tuple_sim:
            log.info('New best model found at epoch {}, prev train unseen tuple sim {:.4f}, new {:.4f}'.format(
                epoch_index, best_train_unseen_tuple_sim, result['train_unseen_tuple_sim']))
            best_train_unseen_tuple_sim = result['train_unseen_tuple_sim']

            # save model for new best
            save_checkpoint(policy, osp.join(args.exp_dir, 'results', 'model_best.pth'),
                            args.use_ddp, args.ddp_rank, num_gpu_per_node)

        # save model for cut-lr epochs
        if args.save_every_epoch or (epoch_index + 1 in args.step_at) or epoch_index == args.warmup_epochs - 1:
            save_checkpoint(policy, osp.join(args.exp_dir, 'results', 'model_epoch_{}.pth'.format(epoch_index)),
                            args.use_ddp, args.ddp_rank, num_gpu_per_node)

    # save model
    save_checkpoint(policy, osp.join(args.exp_dir, 'results', 'model_final.pth'),
                    args.use_ddp, args.ddp_rank, num_gpu_per_node)


def save_checkpoint(model, fname, use_ddp, ddp_rank, num_gpu_per_node):
    if (not use_ddp) or (use_ddp and ddp_rank % num_gpu_per_node == 0):
        os.makedirs(osp.dirname(fname), exist_ok=True)
        torch.save(model.state_dict(), fname)
        log.info('Model saved to {}'.format(fname))

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
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


def main(args):
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.exp_dir))
    os.system('cp -r *.py models {}'.format(args.exp_dir))

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    num_gpu_per_node = torch.cuda.device_count()
    log.info('Found {} gpus on current node'.format(num_gpu_per_node))
    assert num_gpu_per_node > 0
    if args.use_ddp:
        # spawn num_gpu workers for ddp
        args.ddp_world_size = num_gpu_per_node * args.ddp_world_size
        log.info('Spawn {} main_worker for ddp'.format(num_gpu_per_node))
        mp.spawn(main_worker, nprocs=num_gpu_per_node, args=(num_gpu_per_node, args))
    else:
        # simply call main_worker function
        log.info('No data parallelism is used, we call main_worker directly')
        main_worker(0, num_gpu_per_node, args)

    # finished, create empty file thus others could check whether or not this task is done
    open(osp.join(args.exp_dir, 'done'), 'a').close()


if __name__ == '__main__':
    xargs = parse_args()

    xargs.exp_dir = osp.join(xargs.exp_dir, get_random_dir_name())
    os.makedirs(xargs.exp_dir, exist_ok=True)

    # set log file, and import glog after that (since we might change sys.stdout/stderr on set_log_file())
    set_log_file(osp.join(xargs.exp_dir, 'run.log'), file_only=xargs.ssh)

    # do the business
    main(xargs)

