import numpy as np
import torch
#from biggan_generator import BigGANGenerator
from QEBATangentAttack.generator.ae_generator import AEGenerator
from QEBATangentAttack.generator.dct_generator import DCTGenerator
from QEBATangentAttack.generator.gan_generator import GANGenerator
from QEBATangentAttack.generator.nn_generator import NNGenerator
from QEBATangentAttack.generator.pca_generator import PCAGenerator
from QEBATangentAttack.generator.resize_generator import ResizeGenerator
from QEBATangentAttack.generator.unet_generator import UNet
from QEBATangentAttack.generator.vae_generator import VAEGenerator


def load_pgen(dataset, pgen_type, args):
    if dataset == 'ImageNet' or dataset == 'CelebA':
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize9408' or pgen_type == "resize": # FIXME I add "resize"
            p_gen = ResizeGenerator(factor=4.0)
        elif pgen_type == 'DCT2352':
            p_gen = DCTGenerator(factor=8.0)
        elif pgen_type == 'DCT4107':
            p_gen = DCTGenerator(factor=6.0)
        elif pgen_type == 'DCT9408':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'DCT16428':
            p_gen = DCTGenerator(factor=3.0)
        elif pgen_type == 'NNGen':
            p_gen = NNGenerator(N_b=30, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_30_imagenet.model'))
        elif pgen_type == 'ENC':
            p_gen = UNet(n_channels=3)
            p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
        elif pgen_type == 'PCA1000':
            p_gen = PCAGenerator(N_b=1000, approx=True)
            p_gen.load('pca_gen_1000_imagenet.npy')
        elif pgen_type == 'PCA5000':
            p_gen = PCAGenerator(N_b=5000, approx=True)
            p_gen.load('pca_gen_5000_imagenet.npy')
        elif pgen_type == 'PCA9408':
            p_gen = PCAGenerator(N_b=9408, approx=True)
            p_gen.load('pca_gen_9408_imagenet_avg.npy')
        elif pgen_type == 'PCA2352basis':
            p_gen = PCAGenerator(N_b=2352, approx=True, basis_only=True)
            p_gen.load('pca_gen_2352_imagenet_avg.npy')
        elif pgen_type == 'PCA4107basis':
            p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
            p_gen.load('pca_gen_4107_imagenet_avg.npy')
        elif pgen_type == 'PCA4107basismore':
            p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
            p_gen.load('pca_gen_4107_imagenet_rndavg.npy')
        elif pgen_type == 'PCA9408basis':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_avg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'PCA9408basismore':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_rndavg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'PCA9408basisnormed':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_normed_avg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'AE9408':
            p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('ae_generator.model'))
        elif pgen_type == 'GAN128':
            p_gen = GANGenerator(n_z=128, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('gan128_generator.model'))
        elif pgen_type == 'VAE9408':
            p_gen = VAEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('vae_generator.model'))
    elif dataset.startswith('CIFAR'):
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize768' or pgen_type == "resize": # FIXME I add "resize"
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT192':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'DCT300':
            p_gen = DCTGenerator(factor=3.0)
        elif pgen_type == 'DCT768':
            p_gen = DCTGenerator(factor=2.0)
        elif pgen_type == 'DCT1200':
            p_gen = DCTGenerator(factor=1.6)
        elif pgen_type == 'PCA192':
            p_gen = PCAGenerator(N_b=192)
            p_gen.load('pca_gen_192_cifar_avg.npy')
        elif pgen_type == 'PCA192train':
            p_gen = PCAGenerator(N_b=192)
            p_gen.load('pca_gen_192_cifartrain_avg.npy')
        elif pgen_type == 'PCA300train':
            p_gen = PCAGenerator(N_b=300)
            p_gen.load('pca_gen_300_cifartrain_avg.npy')
        elif pgen_type == 'PCA768':
            p_gen = PCAGenerator(N_b=768)
            p_gen.load('pca_gen_768_cifar_avg.npy')
        elif pgen_type == 'PCA768train':
            p_gen = PCAGenerator(N_b=768)
            p_gen.load('pca_gen_768_cifartrain_avg.npy')
        elif pgen_type == 'PCA1200':
            p_gen = PCAGenerator(N_b=1200)
            p_gen.load('pca_gen_1200_cifar_avg.npy')
        elif pgen_type == 'PCA1200train':
            p_gen = PCAGenerator(N_b=1200)
            p_gen.load('pca_gen_1200_cifartrain_avg.npy')
        elif pgen_type == 'PCA2000':
            p_gen = PCAGenerator(N_b=2000)
            p_gen.load('pca_gen_2000_cifar_avg.npy')
        elif pgen_type == 'NNGen50':
            p_gen = NNGenerator(N_b=50, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_50_cifar.model'))
        elif pgen_type == 'NNGen768':
            p_gen = NNGenerator(N_b=768, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_768_cifar.model'))
    return p_gen