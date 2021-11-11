from policy_driven_attack import policy
from policy_driven_attack.models.standard_model import StandardPolicyModel

import math
import numpy as np
import h5py
import dill
import torch
import os.path as osp

def load_weight(model, fname):
    if len(fname) == 0 or not osp.exists(fname):
        raise ValueError('Invalid weight file name: {}'.format(fname))
    if fname.endswith('mnist_carlinet') or fname.endswith('cifar10_carlinet'):
        with h5py.File(fname, 'r') as f:
            for key in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'dense_1', 'dense_2', 'dense_3']:
                m = model.__getattr__(key)
                # weight
                if 'conv' in key:
                    w = np.array(f['model_weights'][key][key]['kernel:0']).transpose(3, 2, 0, 1)
                if 'dense' in key:
                    w = np.array(f['model_weights'][key][key]['kernel:0']).transpose(1, 0)
                assert m.weight.shape == w.shape
                m.weight.data[:] = torch.FloatTensor(w)
                # bias
                b = np.array(f['model_weights'][key][key]['bias:0'])
                assert m.bias.shape == b.shape
                m.bias.data[:] = torch.FloatTensor(b)
    else:
        raw_state_dict = torch.load(fname, map_location='cpu', pickle_module=dill)
        if 'schedule' in raw_state_dict:
            # madry pre-trained model: https://github.com/MadryLab/robustness
            raw_state_dict = raw_state_dict['model']
            state_dict = dict()
            for key, val in raw_state_dict.items():
                if key.startswith('module.model.'):
                    new_key = key.replace('module.model.', '')
                    state_dict[new_key] = val
        elif 'state_dict' in raw_state_dict:
            # pytorch-classification scripts trained model
            raw_state_dict = raw_state_dict['state_dict']
            state_dict = dict()
            for key, val in raw_state_dict.items():
                new_key = key.replace('module.', '')
                state_dict[new_key] = val
        elif max([k.startswith('net.') for k in raw_state_dict.keys()]):
            # train_victim_model.py trained model, or train_grad_model.py trained model
            state_dict = dict()
            for key, val in raw_state_dict.items():
                new_key = key.replace('net.', '')
                state_dict[new_key] = val
        elif max([k.startswith('module.net.') for k in raw_state_dict.keys()]):
            # train_victim_model.py trained model, or train_grad_model.py trained model, with ddp
            state_dict = dict()
            for key, val in raw_state_dict.items():
                new_key = key.replace('module.net.', '')
                state_dict[new_key] = val
        else:
            raise NotImplementedError('Model {} is trained by unknown scripts'.format(fname))
        model.load_state_dict(state_dict)

def make_policy_model(dataset, arch, input_size=0, **kwargs):
    assert dataset in ['debug', 'mnist01', 'mnist', 'cifar10', 'imagenet']

    # scratch or not
    if 'weight_fname' in kwargs:
        weight_fname = kwargs['weight_fname']
        assert osp.exists(weight_fname)

        # all other params in kwargs will be passed into construction function of networks
        del kwargs['weight_fname']
    else:
        weight_fname = None

    if dataset == 'debug':
        if arch == 'empty':
            model = policy.debug.empty(**kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))
        model.mean = [0]
        model.std = [1]
        model.input_space = 'RGB'
        model.input_range = [0, 1]
        model.input_size = [2]
    elif dataset == 'mnist01' or dataset == 'mnist':
        if input_size == 0:
            input_size = 28
        if arch == 'empty':
            model = policy.mnist.empty(input_size=input_size, **kwargs)
        elif arch == 'unet':
            model = policy.mnist.unet(input_size=input_size, **kwargs)
        elif arch == 'carlinet_inv':
            model = policy.mnist.carlinet_inv(input_size=input_size, **kwargs)
            # weight_fname = 'data/mnist_carlinet'
        elif arch == 'vgg11_inv':
            model = policy.mnist.vgg11_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg13_inv':
            model = policy.mnist.vgg13_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg16_inv':
            model = policy.mnist.vgg16_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg19_inv':
            model = policy.mnist.vgg19_inv(input_size=input_size, **kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))

        model.mean = [0.1307]
        model.std = [0.3081]
        model.input_space = 'GRAY'
        model.input_range = [0, 1]
        model.input_size = [1, 28, 28]
    elif dataset == 'cifar10':
        if input_size == 0:
            input_size = 32
        if arch == 'empty':
            model = policy.cifar.empty(input_size=input_size, **kwargs)
        elif arch == 'unet':
            model = policy.cifar.unet(input_size=input_size, **kwargs)
        elif arch == 'carlinet_inv':
            model = policy.cifar.carlinet_inv(input_size=input_size, **kwargs)
            # weight_fname = 'data/cifar10_carlinet'
        elif arch == 'vgg11_inv':
            model = policy.cifar.vgg11_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg13_inv':
            model = policy.cifar.vgg13_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg16_inv':
            model = policy.cifar.vgg16_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg19_inv':
            model = policy.cifar.vgg19_inv(input_size=input_size, **kwargs)
        elif arch == 'resnet20_inv':
            model = policy.cifar.resnet20_inv(input_size=input_size, **kwargs)
        elif arch == 'resnet32_inv':
            model = policy.cifar.resnet32_inv(input_size=input_size, **kwargs)
        elif arch == 'wrn_28_10_drop_inv':
            model = policy.cifar.wrn_28_10_drop_inv(input_size=input_size, **kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))

        if arch == 'carlinet_inv':
            model.mean = [0.5, 0.5, 0.5]
            model.std = [1, 1, 1]
        else:
            model.mean = [0.4914, 0.4822, 0.4465]
            model.std = [0.2023, 0.1994, 0.2010]
        model.input_space = 'RGB'
        model.input_range = [0, 1]
        model.input_size = [3, 32, 32]
    elif dataset == 'imagenet':
        if input_size == 0:
            input_size = 224
        if arch == 'empty':
            model = policy.imagenet.empty(input_size=input_size, **kwargs)
        elif arch == 'vgg11_inv':
            model = policy.imagenet.vgg11_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg13_inv':
            model = policy.imagenet.vgg13_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg16_inv':
            model = policy.imagenet.vgg16_inv(input_size=input_size, **kwargs)
        elif arch == 'vgg19_inv':
            model = policy.imagenet.vgg19_inv(input_size=input_size, **kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))

        model.mean = [0.485, 0.456, 0.406]
        model.std = [0.229, 0.224, 0.225]
        model.input_space = 'RGB'
        model.input_range = [0, 1]
        model.input_size = [3, input_size, input_size]
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

    # load weight if specified
    if weight_fname is not None:
        load_weight(model, weight_fname)

        # still set normal_logstd according to params instead of weight file
        if 'init_std' in kwargs:
            assert model.normal_logstd.numel() == 1
            model.normal_logstd.data[:] = math.log(kwargs['init_std'])

    # return a warped standard model
    return StandardPolicyModel(dataset=dataset, arch=model)
