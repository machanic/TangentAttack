"""Loads a pretrained model, then attacks it.

   This script minimizes the distance of a misclassified image to an original,
   and enforces misclassification with a log barrier penalty.
"""
import argparse
import os, sys

import numpy as np
import pandas as pd
import pickle as pk

import torch
from torch import nn
from torch.autograd import grad


def Top1Criterion(x,y, model):
    """Returns True if model prediction is in top1"""
    return model(x).topk(1)[1].view(-1)==y

def Top5Criterion(x,y, model):
    """Returns True if model prediction is in top5"""
    return (model(x).topk(5)[1]==y.view(-1,1)).any(dim=-1)

def initialize(x,y,criterion,max_iters=1e3, bounds=(0,1)):
    """Generates random perturbations of clean images until images have incorrect label.

    If the image is already mis-classified, then it is not perturbed."""
    xpert = x.clone()
    dt = 0.01

    correct = criterion(xpert,y)
    k=0
    while correct.sum()>0:
        l = correct.sum()
        xpert[correct] = x[correct] + (1.01)**k*dt*torch.randn(l,*xpert.shape[1:],device=xpert.device)
        xpert[correct].clamp_(*bounds)
        correct = criterion(xpert,y)

        k+=1
        if k>max_iters:
            raise ValueError('failed to initialize: maximum iterations reached')

    return xpert

class LogBarrierAttack(object):

    def __init__(self, model, criterion=Top1Criterion, initialize=initialize, norm=2,
            verbose=True, **kwargs):
        """Attack a model using the log barrier constraint to enforce mis-classification.

        Arguments:
            model: PyTorch model, takes batch of inputs and returns logits
            criterion: function which takes in images and labels and a model, and returns
                a boolean vector, which is True is model prediction is correct
                For example, the Top1 or Top5 classification criteria
            initialize (optional): function which takes in images, labels and a model, and
                returns mis-classified images (an initial starting guess) (default: clipped Gaussians)
            norm (optional): norm to measure adversarial distance with (default: 2)
            verbobose (optional): if True (default), display status during attack

        Keyword arguments:
            bounds: tuple, image bounds (default (0,1))
            dt: step size (default: 0.01)
            alpha: initial Lagrange multiplier of log barrier penalty (default: 0.1)
            beta: shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)
            gamma: back track parameter (default: 0.5)
            max_outer: maximum number of outer loops (default: 15)
            tol: inner loop stopping criteria (default: 1e-6)
            max_inner: maximum number of inner loop iterations (default: 500)
            T: softmax temperature in L-infinity norm approximation (default: 500)

        Returns:
            images: adversarial images mis-classified by the model
        """
        super().__init__()
        self.model = model
        self.criterion = lambda x, y: criterion(x,y,model)
        self.initialize = initialize
        self.labels = None
        self.original_images = None
        self.perturbed_images = None

        if not (norm==2 or norm==np.inf):
            raise ValueError('norm must be either 2 or np.inf')
        self.norm = norm
        self.verbose = verbose


        config = {'bounds':(0,1),
                  'dt': 0.01,
                  'alpha':0.1,
                  'beta':0.75,
                  'gamma':0.5,
                  'max_outer':15,
                  'tol':1e-6,
                  'max_inner':int(5e2),
                  'T':500.}
        config.update(kwargs)

        self.hyperparams = config


    def __call__(self, x, y):
        self.labels = y
        self.original_images = x

        config = self.hyperparams
        model = self.model
        criterion = self.criterion

        bounds, dt, alpha0, beta, gamma, max_outer, tol, max_inner, T = (
                config['bounds'], config['dt'], config['alpha'], config['beta'],
                config['gamma'], config['max_outer'], config['tol'], config['max_inner'],
                config['T'])


        Nb = len(y)
        ix = torch.arange(Nb, device=x.device)

        imshape = x.shape[1:]
        PerturbedImages = torch.full(x.shape,np.nan, device=x.device)

        mis0 = criterion(x,y)

        xpert = initialize(x,y,criterion)

        xpert[~mis0]= x[~mis0]
        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,),np.inf,device=x.device)
        xpert.requires_grad_(True)

        for k in range(max_outer):
            alpha = alpha0*beta**k

            diff = (xpert - x).view(Nb,-1).norm(self.norm,-1)
            update= diff>0
            for j in range(max_inner):
                p = model(xpert).softmax(dim=-1)

                pdiff = p.max(dim=-1)[0] - p[ix,y]
                s = -torch.log(pdiff).sum()
                g = grad(alpha*s,xpert)[0] # TODO: use only one grad when norm==Linf
                if self.norm==2:
                    with torch.no_grad():
                        xpert[update] = xpert[update].mul(1-dt).add(-dt,
                                g[update]).add(dt,x[update]).clamp_(*bounds)
                elif self.norm==np.inf:
                    Nb_ = xpert[update].shape[0]
                    xpert_, x_ = xpert[update].view(Nb_,-1), x[update].view(Nb_,-1)
                    z_ = (xpert_ - x_)
                    z = torch.abs(z_)

                    #smooth approximation of Linf norm
                    ex_ = ((z*T).softmax(dim=-1)*z).sum(dim=-1)

                    ginf = grad(ex_.sum(),xpert_)[0]

                    with torch.no_grad():
                        GradientStep = ginf.view(Nb_,*imshape) + g[update]
                        xpert[update] = xpert[update].add(-dt,GradientStep).clamp(*bounds)

                with torch.no_grad():
                    # backtrack
                    c = criterion(xpert,y)
                    while c.any():
                        xpert.data[c] = xpert.data[c].clone().mul(1-gamma).add(gamma,xold[c])
                        c = criterion(xpert,y)

                    diff = (xpert - x).view(Nb,-1).norm(self.norm,-1)
                    boolDiff = diff <= diffBest
                    xbest[boolDiff] = xpert[boolDiff]
                    diffBest[boolDiff] = diff[boolDiff]

                    iterdiff = (xpert - xold).view(Nb,-1).norm(self.norm,-1)
                    #med = diff.median()

                    xold = xpert.clone()


                if self.verbose:
                    sys.stdout.write('  [%2d outer, %4d inner] median & max distance: (%4.4f, %4.4f)\r'
                        %(k, j, diffBest.median() , diffBest.max()))

                if not iterdiff.abs().max()>tol:
                    break

        if self.verbose:
            sys.stdout.write('\n')

        switched = ~criterion(xbest,y)
        PerturbedImages[switched] = xbest.detach()[switched]

        self.perturbed_images = PerturbedImages

        return PerturbedImages

def get_parse_args():
    parser = argparse.ArgumentParser('Attack an example CIFAR10 model with the Log Barrier attack.'
                                      'Writes adversarial distances (and optionally images) to a npz file.')

    groups0 = parser.add_argument_group('Required arguments')
    groups0.add_argument('--data-dir', type=str, required=True,
            help='Directory where CIFAR10 data is saved')

    groups2 = parser.add_argument_group('Optional attack arguments')
    groups2.add_argument('--num-images', type=int, default=1000,metavar='N',
            help='total number of images to attack (default: 1000)')
    groups2.add_argument('--batch-size', type=int, default=200,metavar='N',
            help='number of images to attack at a time (default: 200) ')
    groups2.add_argument('--save-images', action='store_true', default=False,
            help='save perturbed images to a npy file (default: False)')
    groups2.add_argument('--norm', type=str, default='L2',metavar='NORM',
            choices=['L2','Linf'],
            help='The norm measuring distance between images. (default: "L2")')

    groups2.add_argument('--seed', type=int, default=0,
            help='seed for RNG (default: 0)')
    groups2.add_argument('--random-subset', action='store_true',
            default=False, help='use random subset of test images (default: False)')

    group1 = parser.add_argument_group('Attack hyperparameters')
    group1.add_argument('--dt', type=float, default=0.01, help='step size (default: 0.01)')
    group1.add_argument('--alpha', type=float, default=0.1,
            help='initial Lagrange multiplier of log barrier penalty (default: 0.1)')
    group1.add_argument('--beta', type=float, default=0.75,
            help='shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)')
    group1.add_argument('--gamma', type=float, default=0.5,
            help='back track parameter (default: 0.5)')
    group1.add_argument('--max-outer', type=int, default=15,
            help='maximum number of outer loops (default: 15)')
    group1.add_argument('--max-inner', type=int, default=500,
            help='max inner loop iterations (default: 500)')
    group1.add_argument('--tol', type=float, default=1e-6,
            help='inner loop stopping criterion (default: 1e-6)')
    group1.add_argument('--T', type=float, default=500,
            help='softmax temperature for approximating Linf-norm (default: 500)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args

if __name__ == "__main__":
    args = get_parse_args()
    i = 0
    while os.path.exists('attack%s' % i):
        i += 1
    pth = os.path.join('./', 'attack%s/' % i)
    os.makedirs(pth, exist_ok=True)

    args_file_path = os.path.join(pth, 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    has_cuda = torch.cuda.is_available()

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor()])
    ds = CIFAR10(args.data_dir, download=True, train=False, transform=transform)

    if args.random_subset:
        Ix = np.random.choice(10000, size=args.num_images, replace=False)
        Ix = torch.from_numpy(Ix)
    else:
        Ix = torch.arange(args.num_images)  # Use the first N images of test set

    subset = Subset(ds, Ix)

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=has_cuda)

    # Retrieve pre trained model
    classes = 10
    model = ResNeXt34_2x32()
    model.load_state_dict(torch.load('models/example-resnext34_2x32.pth.tar', map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    criterion = lambda x, y: Top1Criterion(x, y, model)
    if args.norm == 'L2':
        norm = 2
    elif args.norm == 'Linf':
        norm = np.inf

    if has_cuda:
        model = model.cuda()

    params = {'bounds': (0, 1),
              'dt': args.dt,
              'alpha': args.alpha,
              'beta': args.beta,
              'gamma': args.gamma,
              'max_outer': args.max_outer,
              'tol': args.tol,
              'max_inner': args.max_inner,
              'T': args.T}

    attack = Attack(model, norm=norm, **params)

    d1 = torch.full((args.num_images,), np.inf)
    d2 = torch.full((args.num_images,), np.inf)
    dinf = torch.full((args.num_images,), np.inf)
    if has_cuda:
        d1 = d1.cuda()
        d2 = d2.cuda()
        dinf = dinf.cuda()

    if args.save_images:
        PerturbedImages = torch.full((args.num_images, 3, 32, 32), np.nan)
        labels = torch.full((args.num_images,), -1, dtype=torch.long)
        if has_cuda:
            PerturbedImages = PerturbedImages.cuda()
            labels = labels.cuda()

    K = 0
    for i, (x, y) in enumerate(loader):
        print('Batch %2d/%d:' % (i + 1, len(loader)))

        Nb = len(y)
        if has_cuda:
            x, y = x.cuda(), y.cuda()

        xpert = attack(x, y)

        diff = x - xpert.detach()
        l1 = diff.view(Nb, -1).norm(p=1, dim=-1)
        l2 = diff.view(Nb, -1).norm(p=2, dim=-1)
        linf = diff.view(Nb, -1).norm(p=np.inf, dim=-1)

        ix = torch.arange(K, K + Nb, device=x.device)

        if args.save_images:
            PerturbedImages[ix] = xpert
            labels[ix] = y
        d1[ix] = l1
        d2[ix] = l2
        dinf[ix] = linf

        K += Nb

    if args.norm == 'L2':
        md = d2.median()
        mx = d2.max()
    elif args.norm == 'Linf':
        md = dinf.median()
        mx = dinf.max()

    print('\nDone. Statistics in %s norm:' % args.norm)
    print('  Median adversarial distance: %.3g' % md)
    print('  Max adversarial distance:    %.3g' % mx)

    st = 'logbarrier-' + args.norm

    dists = {'index': Ix.cpu().numpy(),
             'l1': d1.cpu().numpy(),
             'l2': d2.cpu().numpy(),
             'linf': dinf.cpu().numpy()}
    if args.save_images:
        dists['perturbed'] = PerturbedImages.cpu().numpy()
        dists['labels'] = labels.cpu().numpy()

    with open(os.path.join(pth, st + '.npz'), 'wb') as f:
        np.savez(f, **dists)

    # if __name__=="__main__":
    #    main()
