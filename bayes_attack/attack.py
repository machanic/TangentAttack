import json
import os
import sys
sys.path.append(os.getcwd())
import random
import argparse 
from collections import defaultdict, OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from bayes_attack.utils import proj, latent_proj, transform

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import joint_optimize, gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch, get_best_candidates
import os.path as osp
import glog as log

from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel


class BayesAttack(object):
    def __init__(self, model, dataset, image_height, image_width, channels,  dim, norm, epsilon, discrete, hard_label,optimize_acq,
                 standardize,standardize_every_iter, q, num_restarts, initial_samples, iter, acqf, beta):
        self.model = model
        self.dataset = dataset
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.epsilon = epsilon
        self.discrete = discrete
        self.norm = norm
        self.hard_label = hard_label
        self.dim = dim
        self.latent_dim = dim * dim * 2 * channels
        self.standardize = standardize
        self.standardize_every_iter = standardize_every_iter
        self.q = q
        self.num_restarts = num_restarts
        # self.bounds = torch.tensor([[-2.0] * self.latent_dim, [2.0] * self.latent_dim]).float().cuda()、
        self.bounds = torch.tensor([[0] * self.latent_dim, [1.0] * self.latent_dim]).float().cuda()
        self.initial_samples = initial_samples
        self.iter = iter
        self.acqf = acqf
        self.beta = beta
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, 1)
        self.total_images = len(self.dataset_loader.dataset)
        self.maximum_queries = args.max_queries
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.optimize_acq = optimize_acq

    def obj_func(self, x, x0, y0):
        x = transform(x, self.dataset,self.image_height, self.image_width, self.channels, self.dim).cuda()
        x = proj(x, self.epsilon, self.norm, self.discrete)
        y = self.model((x + x0).cuda().float())
        if self.hard_label:
            index = torch.argmax(y, dim=1)
            f = torch.where(index == y0, torch.ones_like(index),
                            torch.zeros_like(index)).float()  # 错误分类返回0，正确分类返回-1，return -f
        else:
            y = torch.log_softmax(y, dim=1)
            max_score = y[:, y0]
            y, index = torch.sort(y, dim=1, descending=True)
            select_index = (index[:, 0] == y0).long()
            next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
            f = torch.max(max_score - next_max, torch.zeros_like(max_score))
        return -f

    def initialize_model(self, x0, y0, n=5):
        # initialize botorch GP model
        # generate prior xs and ys for GP
        train_x = 2 * torch.rand(n, self.latent_dim).float().cuda() - 1
        if self.norm == "linf":
            train_x = latent_proj(train_x, self.epsilon)
        train_obj = self.obj_func(train_x, x0, y0)
        mean, std = train_obj.mean(), train_obj.std()
        if self.standardize:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
        best_observed_value = train_obj.max().cuda()  # item() --> cuda() FIXME
        # define models for objective and constraint
        model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
        model = model.to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = mll.to(train_x)  # loss function  , it is set mll and all submodules to the specified dtype and device
        return train_x, train_obj, mll, model, best_observed_value, mean, std

    def optimize_acqf_and_get_observation(self, acq_func, x0, y0):
        # Optimizes the acquisition function, returns new candidate new_x
        # and its objective function value new_obj

        if self.optimize_acq == 'scipy':
            candidates = joint_optimize(
                acq_function=acq_func,
                bounds=self.bounds,
                q=self.q,
                num_restarts=self.num_restarts,
                raw_samples=200
            )
        else:
            # gen_batch_initial_conditions: Generate a batch of initial conditions for random-restart optimziation
            Xinit = gen_batch_initial_conditions(
                acq_func,  # The acquisition function to be optimized.
                self.bounds,  # bounds: A 2 x d tensor of lower and upper bounds for each column of X.
                q=self.q,  # The number of candidates to consider.
                num_restarts=self.num_restarts,
                # The number of starting points for multistart acquisition function optimization.
                raw_samples=500  # The number of raw samples to consider in the initialization heuristic.
            )

            #  Generate a set of candidates using a torch.optim optimizer.
            # Optimizes an acquisition function starting from a set of initial candidates using an optimizer from torch.optim.
            batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=Xinit,  # Starting points for optimization.
                acquisition_function=acq_func,  # Acquisition function to be used
                lower_bounds=self.bounds[0],
                upper_bounds=self.bounds[1],
                verbose=False
            )
            # Extract best (q-batch) candidate from batch of candidates
            candidates = get_best_candidates(batch_candidates, batch_acq_values)
        # A tensor of size q x d (if q-batch mode) or d from batch_candidates with the highest associated value.
        # observe new values
        new_x = candidates[0].detach()
        if self.norm == "linf":
            new_x = latent_proj(new_x, self.epsilon)
        new_obj = self.obj_func(new_x, x0, y0)

        return new_x, new_obj

    def calculate_distortion(self, x_adv, x_original, query, image_index):
        ord = np.inf if self.norm == "linf" else 2
        dist = torch.norm((x_adv - x_original).view(x_adv.size(0), -1), ord, 1).item()
        self.distortion_all[image_index][query] = dist
        return dist

    def bayes_opt(self, image_index, x0, y0, args=None):
        """
        Main Bayesian optimization loop. Begins by initializing model, then for each
        iteration, it fits the GP to the data, gets a new point with the acquisition
        function, adds it to the dataset, and exits if it's a successful attack
        """
        x0 = x0.cuda()
        best_observed = []
        query, success = 0, 0
        # call helper function to initialize model
        train_x, train_obj, mll, gp_model, best_value, mean, std = self.initialize_model(
            x0, y0, n=self.initial_samples)
        if self.standardize_every_iter:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
        best_observed.append(best_value.item())
        query += self.initial_samples
        adv_image = x0.clone()
        # run args.iter rounds of BayesOpt after the initial random batch
        for iteration in range(self.iter):

            # fit the model
            fit_gpytorch_model(mll)  # MarginalLogLikelihood to be maximized

            # define the qNEI acquisition module using a QMC sampler
            if self.q != 1:
                qmc_sampler = SobolQMCNormalSampler(num_samples=2000, seed=args.seed)
                qEI = qExpectedImprovement(model=gp_model, sampler=qmc_sampler, best_f=best_value)  # GP model 用在这了
            else:
                # Internally, q-batch acquisition functions operate on input tensors of shape b×q×d, where b is the number of t-batches, q is the number of design points to be considered concurrently, and d is the dimension of the parameter space. Their output is a one-dimensional tensor with b elements, with the i-th element corresponding to the i-th t-batch. Always requiring explicit t-batch and q-batch dimensions makes it easier and less ambiguous to work with samples from the posterior in a consistent fashion.
                if self.acqf == 'EI':
                    qEI = ExpectedImprovement(model=gp_model, best_f=best_value)  # model 用在这了
                elif self.acqf == 'PM':
                    qEI = PosteriorMean(gp_model)
                elif self.acqf == 'POI':
                    qEI = ProbabilityOfImprovement(gp_model, best_f=best_value)
                elif self.acqf == 'UCB':
                    qEI = UpperConfidenceBound(gp_model, beta=self.beta)

            # optimize and get new observation
            new_x, new_obj = self.optimize_acqf_and_get_observation(qEI, x0, y0)

            if self.standardize:
                new_obj = (new_obj - mean) / std
            # update training points
            train_x = torch.cat((train_x, new_x))
            train_obj = torch.cat((train_obj, new_obj))
            if self.standardize_every_iter:
                train_obj = (train_obj - train_obj.mean()) / train_obj.std()

            # update progress
            best_value, best_index = train_obj.max(0)
            best_observed.append(best_value.item())
            best_candidate = train_x[best_index]

            # reinitialize the model so it is ready for fitting on next iteration
            torch.cuda.empty_cache()
            gp_model.set_train_data(train_x, train_obj, strict=False)

            # get objective value of best candidate; if we found an adversary, exit
            best_candidate = best_candidate.view(1, -1)
            best_candidate = transform(best_candidate, self.dataset, self.image_height,self.image_width, self.channels, self.dim).cuda()
            best_candidate = proj(best_candidate, self.epsilon, self.norm, self.discrete)
            adv_image = best_candidate + x0
            adv_label = torch.argmax(self.model(adv_image.cuda().float()),dim=1).item()
            query += self.q  # FIXME add to here
            dist = self.calculate_distortion(adv_image, x0, query, image_index)
            log.info("{}-th image {}-th iter, dist {:.4f} query: {}".format(image_index, iteration, dist, query))
            if adv_label != y0:
                success = 1
                if self.norm == "linf":
                    log.info('Success! Adversarial predicted label {}, norm: {}, query: {}'.format(adv_label.item(), best_candidate.abs().max().item(),query))
                else:
                    log.info('Success! Adversarial predicted label {}, norm: {}, query: {}'.format(adv_label.item(), best_candidate.norm().item(),query))
                break   #
            # query += self.q  # FIXME delete!
            if query > self.maximum_queries:
                log.info('query is {} out of maximum of {} queries!'.format(query, self.maximum_queries))
                break
        # not successful (ran out of query budget)
        return adv_image, torch.tensor([query]).float(), torch.tensor([dist]).float(), torch.tensor([success]).float()

    def attack_all_images(self, args, arch_name, result_dump_path):

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            selected = torch.arange(batch_index, min(batch_index + 1, self.total_images))

            adv_images, query, distortion_with_max_queries, success = self.bayes_opt(batch_index, images, true_labels[0].item(), args)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            with torch.no_grad():
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            not_done = not_done * adv_pred.eq(true_labels.cuda()).float()
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success.float() *(query <= self.maximum_queries).float()
            success_query = success * query
            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset,  norm, attack_defense):
    target_str = "untargeted"
    if attack_defense:
        dirname = 'BayesAttack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'BayesAttack-{}-{}-{}'.format(dataset, norm, target_str)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=None, type=str, help='network architecture') # network architecture (resnet50, vgg16_bn, or inception_v3)
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--json-config', type=str, default='./configures/Bayes.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--acqf', type=str, default='EI') # BayesOpt acquisition function
    parser.add_argument('--beta', type=float, default=1.0) # hyperparam for UCB acquisition function
    parser.add_argument('--dim', type=int, default=18) # dimension of attack
    parser.add_argument('--dataset', type=str, required=True) # dataset to attack
    parser.add_argument('--discrete',  action='store_true') # if True, project to boundary of epsilon ball (instead of just projecting inside)
    parser.add_argument('--epsilon', type=float) # bound on perturbation norm
    parser.add_argument('--hard_label', action='store_true') # hard-label vs soft-label attack
    parser.add_argument('--iter', type=int, default=10000) # number of BayesOpt iterations to perform
    parser.add_argument('--initial_samples', type=int, default=5) # number of samples taken to form the GP prior
    parser.add_argument('--norm', required=True,type=str,choices=["l2","linf"]) # perform L_inf norm attack
    parser.add_argument('--num_restarts', type=int, default=1) # hyperparam for acquisition function
    parser.add_argument('--q', type=int, default=1) # number of candidates to receive from acquisition function
    parser.add_argument('--standardize', default=False, action='store_true') # normalize objective values
    parser.add_argument('--standardize_every_iter', default=False, action='store_true')  # normalize objective values at every BayesOpt iteration
    parser.add_argument('--seed', type=int, default=1) # random seed
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--optimize_acq', type=str,
                        default='scipy')  # backend for acquisition function optimization (torch or scipy)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.hard_label == True
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)

    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.norm, args.attack_defense))  # 随机产生一个目录用于实验
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
        attacker = BayesAttack(model, args.dataset, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                     args.dim, args.norm, args.epsilon, args.discrete,args.hard_label,args.optimize_acq,
                               args.standardize, args.standardize_every_iter,
                               args.q, args.num_restarts,args.initial_samples,args.iter,args.acqf,args.beta)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
