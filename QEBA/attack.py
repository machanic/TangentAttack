from __future__ import print_function
from __future__ import division
import os
import sys
sys.path.append(os.getcwd())
import argparse
import json
import random
import warnings
import time
from collections import defaultdict, OrderedDict
from types import SimpleNamespace

import glog as log
import os.path as osp

from QEBA.adversarial import Adversarial
from QEBA.rv_generator import load_pgen
from QEBA.utils import Misclassification, MSE, TargetClass
import math
import torch
from torch.nn import functional as F
import numpy as np

from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from config import IN_CHANNELS, CLASS_NUM, IMAGE_DATA_ROOT


class QEBA(object):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.
    Notes
    -----
    Features:
    * ability to switch between two types of distances: MSE and Linf.
    * ability to continue previous attacks by passing an instance of the
      Adversarial class
    * ability to pass an explicit starting point; especially to initialize
      a targeted attack
    * ability to pass an alternative attack used for initialization
    * ability to specify the batch size
    """

    def __init__(self, model,  dataset, clip_min, clip_max, height, width, channels, norm, epsilon,
                 iterations=64,
                 initial_num_evals=100,
                 max_num_evals=10000,
                 stepsize_search='geometric_progression',
                 gamma=0.01,
                 batch_size=256,
                 internal_dtype=torch.float64,
                 log_every_n_steps=1,
                 verbose=False,
                 rv_generator=None, atk_level=None,
                 mask=None,
                 save_calls=None,
                 discretize=False,
                 suffix='',
                 plot_adv=True,
                 threshold=None,
                 distance=MSE,
                 maximum_queries=10000
                 ):
        """Applies QEBA
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        iterations : int
            Number of iterations to run.
        initial_num_evals: int
            Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_num_evals: int
            Maximum number of evaluations for gradient estimation.
        stepsize_search: str
            How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma: float
            The binary search threshold theta is gamma / sqrt(d) for
                   l2 attack and gamma / d for linf attack.
        batch_size : int
            Batch size for model prediction. It is not the data_loader's batch size!
            Higher precision might be slower but is numerically more stable.
        log_every_n_steps : int
            Determines verbositity of the logging.
        verbose : bool
            Controls verbosity of the attack.
        """
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm = norm
        self.epsilon = epsilon
        self.ord = np.inf if self.norm == "linf" else 2
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose

        self.internal_dtype = internal_dtype
        self.log_every_n_steps = log_every_n_steps
        self.rv_generator = rv_generator
        self.discretize = discretize
        self.suffix = suffix
        self.plot_adv = plot_adv

        self._default_threshold = threshold
        self._default_distance = distance

        self.iterations = iterations
        self.atk_level = atk_level  # int type

        self.shape = [channels, height, width]
        if mask is not None:
            self.use_mask = True
            self.pert_mask = mask
            self.loss_mask = 1 - mask
        else:
            self.use_mask = False
            self.pert_mask = torch.ones(self.shape).float()
            self.loss_mask = torch.ones(self.shape).float()
        self.__mask_succeed = 0

        # Set binary search threshold.
        self.fourier_basis_aux = None
        self.dim = np.prod(self.shape)
        if self.norm == 'l2':
            self.theta = self.gamma / np.sqrt(self.dim)
        else:
            self.theta = self.gamma / self.dim

        self.printv('QEBA optimized for {} distance'.format(self.norm))
        self.save_calls = save_calls
        if save_calls is not None:
            if not os.path.isdir(save_calls):
                os.mkdir(save_calls)
            self.save_cnt = 0
            self.save_outs = []
            self.save_hashes = []

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, 1)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def gen_random_basis(self, N):
        basis = torch.from_numpy(np.random.randn(N, *self.shape)).type(self.internal_dtype)
        return basis

    def gen_custom_basis(self, N, sample, atk_level=None):
        if self.rv_generator is not None:
            basis = torch.from_numpy(self.rv_generator.generate_ps(sample, N)).type(self.internal_dtype)
        else:
            basis = self.gen_random_basis(N)
        return basis

    def count_stop_query_and_distortion(self, images, perturbed, adversarial, success_stop_queries, batch_image_positions):

        dist = torch.norm((perturbed - images).view(1, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = adversarial._total_prediction_calls
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][adversarial._total_prediction_calls] = dist[
                inside_batch_index].item()


    def attack(self, image_index, a):
        """
        a: Adversarial class
        """
        # query = torch.zeros(1).float()
        success_stop_queries = torch.zeros(1).float()  # stop query count once the distortion < epsilon
        batch_size = a.unperturbed.size(0)
        batch_image_positions = np.arange(image_index * batch_size,
                                          min((image_index + 1) * batch_size, self.total_images)).tolist()

        self.external_dtype = a.unperturbed.dtype
        assert self.internal_dtype in [torch.float32, torch.float64]
        assert self.external_dtype in [torch.float32, torch.float64]
        assert not (self.external_dtype == torch.float64 and
                    self.internal_dtype == torch.float32)
        a.set_distance_dtype(self.internal_dtype)
        # ===========================================================
        # Increase floating point precision
        # Construct batch decision function with binary output.
        # ===========================================================
        def decision_function(x):
            outs = []
            num_batchs = int(math.ceil(x.size(0) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j:
                                  self.batch_size * (j + 1)]
                current_batch = current_batch.type(self.external_dtype)
                out = a.forward(current_batch, strict=False)[1]  # forward function returns predictions, is_adversarial, 这里is_adversarial其实是prediction == true label
                outs.append(out)
            outs = torch.cat(outs, dim=0)
            return outs

        # ===========================================================
        # intialize time measurements
        # ===========================================================
        self.time_gradient_estimation = 0
        self.time_search = 0
        self.time_initialization = 0
        # ===========================================================
        # Initialize variables, constants, hyperparameters, etc.
        # ===========================================================
        warnings.simplefilter('always', UserWarning) # make sure repeated warnings are shown
        # ===========================================================
        # get bounds
        bounds = a.bounds()
        self.clip_min, self.clip_max = bounds
        # ===========================================================
        # Find starting point
        # ===========================================================
        _, num_evals = self.initialize_starting_point(a)
        # query += num_evals
        if a.perturbed is None:
            warnings.warn(
                'Initialization failed. It might be necessary to pass an explicit starting point.')
            return
        # get original and starting point in the right format
        assert a.perturbed.dtype == self.external_dtype
        original = a.unperturbed.type(self.internal_dtype)  # target class image
        perturbed = a.perturbed.type(self.internal_dtype)
        original = original.squeeze()
        if perturbed.dim() > 3:
            perturbed = perturbed.squeeze(0)

        self.count_stop_query_and_distortion(original, perturbed, a, success_stop_queries, batch_image_positions)

        # ===========================================================
        # Iteratively refine adversarial
        # ===========================================================
        # Project the initialization to the boundary.
        perturbed, dist_post_update, mask_succeed, num_evals = self.binary_search_batch(original, torch.unsqueeze(perturbed,dim=0),
                                                                             decision_function)
        # query += num_evals
        dist = torch.norm((perturbed - original).view(batch_size, -1), self.ord, 1)
        self.count_stop_query_and_distortion(original, perturbed, a, success_stop_queries, batch_image_positions)

        # log starting point
        # distance = a.distance.value
        # self.log_step(0, distance, a=a, perturbed=perturbed)
        if mask_succeed > 0:
            self.__mask_succeed = 1
            return
        step = 0
        old_perturbed = perturbed
        while a._total_prediction_calls < self.maximum_queries:
            step += 1
            # ===========================================================
            # Gradient direction estimation.
            # ===========================================================
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)
            c0 = a._total_prediction_calls
            # Choose number of evaluations.
            num_evals = int(min([int(self.initial_num_evals * np.sqrt(step)), self.max_num_evals]))
            # approximate gradient.
            gradf, avg_val = self.approximate_gradient(decision_function, perturbed,
                                                       num_evals, delta, atk_level=self.atk_level)
            # query += num_evals
            # Calculate auxiliary information for the exp
            # grad_gt = a._model.gradient_one(perturbed, label=a._criterion.target_class()) * self.pert_mask
            # dist_dir = original - perturbed
            # if self.rv_generator is not None:
            #     rho = self.rho_ref
            # else:
            #     rho = 1.0

            if self.norm == 'linf':
                update = torch.sign(gradf)
            else:
                update = gradf
            c1 = a._total_prediction_calls
            # ===========================================================
            # Update, and binary search back to the boundary.
            # ===========================================================
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                epsilon, num_evals = self.geometric_progression_for_stepsize(perturbed, update, dist, decision_function, step)
                # query += num_evals
                # Update the sample.
                perturbed = torch.clamp(perturbed + (epsilon * update).type(self.internal_dtype), self.clip_min, self.clip_max)
                c2 = a._total_prediction_calls
                # Binary search to return to the boundary.
                perturbed, dist_post_update, mask_succeed, num_evals = self.binary_search_batch(original, perturbed[None], decision_function)
                # query += num_evals
                c3 = a._total_prediction_calls
                self.count_stop_query_and_distortion(original, perturbed, a, success_stop_queries, batch_image_positions)

            elif self.stepsize_search == 'grid_search':
                # Grid search for stepsize.
                epsilons = torch.logspace(-4, 0, steps=20) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.view(epsilons_shape) * update
                perturbeds = torch.clamp(perturbeds, min=self.clip_min, max=self.clip_max)
                idx_perturbed = decision_function(perturbeds)
                self.count_stop_query_and_distortion(original, perturbed, a, success_stop_queries,
                                                     batch_image_positions)
                if idx_perturbed.sum().item() > 0:
                    # Select the perturbation that yields the minimum distance after binary search.
                    perturbed, dist_post_update, mask_succeed, num_evals = self.binary_search_batch(original, perturbeds[idx_perturbed], decision_function)
                    # query += num_evals
                    self.count_stop_query_and_distortion(original, perturbed, a, success_stop_queries,
                                                         batch_image_positions)
            # compute new distance.
            dist = torch.norm((perturbed - original).view(batch_size, -1), self.ord, 1)
            log.info(
                '{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(image_index + 1, step, self.norm,
                                                                                      dist.item(),
                                                                                      a._total_prediction_calls))
            # ===========================================================
            # Log the step
            # ===========================================================
            # if self.norm == 'l2':
            #     distance = dist ** 2 / self.dim / (self.clip_max - self.clip_min) ** 2
            # elif self.norm == 'linf':
            #     distance = dist / (self.clip_max - self.clip_min)
            # self.log_step(step, distance, a=a, perturbed=perturbed, update=update * epsilon,
            #               aux_info=(gradf, grad_gt, dist_dir, rho))
            if self.stepsize_search == 'geometric_progression':
                self.printv("Call in grad approx / geo progress / binary search: {}/{}/{}".format(c1 - c0, c2 - c1, c3 - c2))
            a.__best_adversarial = perturbed

            if mask_succeed > 0:
                self.__mask_succeed = 1
                break
            if a._total_prediction_calls >= self.maximum_queries:
                break
            old_perturbed = perturbed
        # Save the labels
        if self.save_calls is not None:
            log.info("Total saved calls: {}".format(len(self.save_outs)))

        return old_perturbed, torch.tensor([a._total_prediction_calls]).float(), success_stop_queries, dist, (dist <= self.epsilon)

    def initialize_starting_point(self, a):
        starting_point = self._starting_point
        num_evals = 0
        a.__best_adversarial = starting_point.clone()  # FIXME 我自己添加的
        if a.perturbed is not None:
            log.info('Attack is applied to a previously found adversarial.'
                ' Continuing search for better adversarials.')
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring starting_point parameter because the attack'
                    ' is applied to a previously found adversarial.')
            return a.perturbed, num_evals

        if starting_point is not None:
            a.forward_one(starting_point)
            assert a.perturbed is not None, ('Invalid starting point provided. Please provide a starting point that is adversarial.')
            return a.perturbed, num_evals + 1

        """
        Apply BlendedUniformNoiseAttack if without initialization.
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        while True:
            random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).type(self.external_dtype)
            _, success = a.forward_one(random_noise)
            num_evals += 1
            if success:
                break
            if num_evals > 1e4:  # FIXME replaced with HSJA that uses a target image?
                return

        # Binary search to minimize l2 distance to the original input.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            # FIXME 这个a.unperturbed其实是target class image
            blended = self.loss_mask * ((1 - mid) * a.unperturbed + mid * random_noise) + \
                      (torch.ones_like(self.loss_mask) - self.loss_mask) * a.perturbed
            _, success = a.forward_one(blended.type(self.external_dtype))
            num_evals += 1
            if success:
                high = mid
            else:
                low = mid
        return blended, num_evals

    def compute_distance(self, x_ori, x_pert, norm='l2'):
        # Compute the distance between two images.
        if norm == 'l2':
            return torch.norm((x_ori - x_pert)*self.loss_mask, p=2).item()
        elif norm == 'linf':
            return torch.max(torch.abs(x_ori - x_pert)).item()

    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.min(torch.max(image, clip_min), clip_max)

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        alphas_shape = [alphas.size(0)] + [1] * len(self.shape)
        alphas = alphas.view(*alphas_shape)
        if self.norm == 'l2':
            projected = self.loss_mask * ((1 - alphas) * unperturbed + alphas * perturbed_inputs) + (
                        torch.ones_like(self.loss_mask) - self.loss_mask) * perturbed_inputs
        elif self.norm == 'linf':
            projected = self.clip_image(perturbed_inputs, unperturbed - alphas, unperturbed + alphas)
        return projected

    def binary_search_batch(self, unperturbed, perturbed_inputs,
                            decision_function):
        """ Binary search to approach the boundary. """
        num_evals = 0
        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = torch.tensor(
            [self.compute_distance(unperturbed, perturbed_x, self.norm) for perturbed_x in perturbed_inputs])

        # Choose upper thresholds in binary searchs based on constraint.
        if self.norm == 'linf':
            highs = dists_post_update
            # Stopping criteria.
            thresholds = torch.clamp_max(dists_post_update * self.theta, max=self.theta)
        else:
            highs = torch.ones(perturbed_inputs.size(0))
            thresholds = self.theta

        lows = torch.zeros(perturbed_inputs.size(0))
        lows = lows.type(self.internal_dtype)
        highs = highs.type(self.internal_dtype)
        if self.use_mask:
            _mask = torch.tensor([self.pert_mask] * perturbed_inputs.size(0))
            masked = perturbed_inputs * _mask + unperturbed * (torch.ones_like(_mask) - _mask)
            masked_decisions = decision_function(masked)
            num_evals += masked.size(0)
            highs[masked_decisions == 1] = 0
            succeed = torch.sum(masked_decisions).item() > 0
        else:
            succeed = False
        # Call recursive function.
        while torch.max((highs - lows) / thresholds).item() > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)
            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs)
            num_evals += mid_inputs.size(0)
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = torch.tensor([self.compute_distance(unperturbed, out, self.norm) for out in out_inputs])
        idx = torch.argmin(dists)
        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist, succeed, num_evals

    def select_delta(self, dist_post_update, current_iteration):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.norm == 'l2':
                delta = np.sqrt(self.dim) * self.theta * dist_post_update
            elif self.norm == 'linf':
                delta = self.dim * self.theta * dist_post_update
        return delta

    def approximate_gradient(self, decision_function, sample,
                             num_evals, delta,  atk_level=None):
        """ Gradient direction estimation """
        # import time
        # t0 = time.time()
        dims = tuple(range(1, 1 + len(self.shape)))

        rv_raw = self.gen_custom_basis(num_evals, sample=sample.detach().cpu().numpy(),  atk_level=atk_level)

        _mask = torch.stack([self.pert_mask] * num_evals)
        rv = rv_raw * _mask
        rv = rv / torch.sqrt(torch.sum(torch.mul(rv,rv),dim=dims,keepdim=True))
        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, min=self.clip_min, max=self.clip_max)
        if self.discretize:
            perturbed = (perturbed * 255.0).round() / 255.0
        rv = (perturbed - sample) / delta
        # query the model.
        decisions = decision_function(perturbed)
        # t4 = time.time()
        decision_shape = [decisions.size(0)] + [1] * len(self.shape)
        fval = 2 * decisions.type(self.internal_dtype).view(decision_shape) - 1.0
        # Baseline subtraction (when fval differs)
        vals = fval if torch.abs(torch.mean(fval)).item() == 1.0 else fval - torch.mean(fval).item()
        # vals = fval
        gradf = torch.mean(vals * rv, dim=0)
        # Get the gradient direction.
        gradf = gradf / torch.linalg.norm(gradf)
        return gradf, torch.mean(fval)

    def geometric_progression_for_stepsize(self, x, update, dist,
                                           decision_function,
                                           current_iteration):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        if hasattr(dist,"item"):
            dist = dist.item()
        num_evals = 0
        if self.use_mask:
            size_ratio = np.sqrt(self.pert_mask.sum().item() / torch.numel(self.pert_mask).item())
            epsilon = dist * size_ratio / np.sqrt(current_iteration) + 0.1
        else:
            epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = torch.clamp(x + epsilon * update, min=self.clip_min, max=self.clip_max)
            success = bool(decision_function(updated[None])[0].item())
            num_evals += 1
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover
        return epsilon, num_evals

    def log_step(self, step, distance, message='', always=False, a=None, perturbed=None, update=None, aux_info=None):
        def cos_sim(x1, x2):
            cos = (x1 * x2).sum() / torch.sqrt((x1 ** 2).sum() * (x2 ** 2).sum())
            return cos
        assert len(self.logger) == step
        if aux_info is not None:
            gradf, grad_gt, dist_dir, rho = aux_info
            cos_est = cos_sim(-gradf, grad_gt)
            cos_distpred = cos_sim(dist_dir, -gradf)
            cos_distgt = cos_sim(dist_dir, grad_gt)

            self.logger.append(
                (a._total_prediction_calls, distance, cos_est.item(), rho, cos_distpred.item(), cos_distgt.item()))
        else:
            self.logger.append((a._total_prediction_calls, distance, 0, 0, 0, 0))
        if not always and step % self.log_every_n_steps != 0:
            return
        self.printv('Step {}: {:.5e} {}'.format(
            step,
            distance,
            message))
        if aux_info is not None:
            self.printv("\tEstimated vs. GT: {}".format(cos_est))
            self.printv("\tRho: {}".format(rho))
            self.printv("\tEstimated vs. Distance: {}".format(cos_distpred))
            self.printv("\tGT vs. Distance: {}".format(cos_distgt))
        if not self.plot_adv:
            return  # Dont plot
        if a is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            # plt.imshow(perturbed[:,:,::-1]/255)  #keras
            plt.imshow(perturbed.transpose(1, 2, 0))  # pytorch
            np.savez('QEBA/perturbed%s%d.npz' % (self.suffix, step), pert=perturbed.transpose(1, 2, 0),
                     info=np.array([a._total_prediction_calls, distance]))

            plt.axis('off')
            plt.title('Call %d Distance %f' % (a._total_prediction_calls, distance))
            fig.savefig('QEBA/%sstep%d.png' % (self.suffix, step))
            plt.close(fig)
            if update is not None:
                fig = plt.figure()
                abs_update = (update - update.min()) / (update.max() - update.min())
                plt.imshow(abs_update.transpose(1, 2, 0))  # pytorch
                plt.axis('off')
                plt.title('Call %d Distance %f' % (a._total_prediction_calls, distance))
                fig.savefig('QEBA/update%d.png' % step)
                plt.close(fig)
            #
            self.printv("Call:", a._total_prediction_calls, "Saved to",
                        'QEBA/%sstep%d.png' % (self.suffix, step))

    def printv(self, *args, **kwargs):
        if self.verbose:
            log.info(*args, **kwargs)

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


    def initialize(self, model, sample, decision_function, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = decision_function(random_noise[None])[0].item()
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long()
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                                size=target_labels[invalid_target_index].size()).long()
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = self.get_image_of_target_class(self.dataset_name,target_labels, model).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = decision_function(blended[None])[0].item()
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very very small, this case will cause inifinity loop
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization, num_eval

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and target_model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
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
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item()!=true_labels[0].item()
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = self.get_image_of_target_class(self.dataset_name,target_labels, target_model)[0]
                self._default_criterion = TargetClass(target_labels[0].item())
                a = Adversarial(target_model, self._default_criterion, images, true_labels[0].item(),
                                distance=self._default_distance, threshold=self._default_threshold,
                                targeted_attack=args.targeted)
            else:
                target_labels = None
                self._default_criterion = Misclassification()
                a = Adversarial(target_model, self._default_criterion, images, true_labels[0].item(),
                                distance=self._default_distance, threshold=self._default_threshold,
                                targeted_attack=args.targeted)
                self.external_dtype = a.unperturbed.dtype
                def decision_function(x):
                    out = a.forward(x, strict=False)[1]  # forward function returns pr
                    return out
                target_images,num_calls = self.initialize(target_model, images.squeeze(0),decision_function,None,true_labels,target_labels)


            if model is None or self._default_criterion is None:
                raise ValueError('The attack needs to be initialized'
                                 ' with a model and a criterion or it'
                                 ' needs to be called with an Adversarial'
                                 ' instance.')


            # p_gen = self.rv_generator
            # if p_gen is None:
            #     rho = 1.0
            # else:
            #     loss_ = F.cross_entropy(logit, true_labels.cuda())
            #     loss_.backward()
            #     grad_gt = images.grad.detach()
            #
            #     rho = p_gen.calc_rho(grad_gt, images).item()
            # self.rho_ref = rho

            self._starting_point = target_images # Adversarial input to use as a starting point

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index,a)
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

            # 每攻击成功就写一个
            # meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
            #                   "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
            #                   # "mean_query": self.success_query_all[self.success_all.bool()].mean().item(),
            #                   # "median_query": self.success_query_all[self.success_all.bool()].median().item(),
            #                   # "max_query": self.success_query_all[self.success_all.bool()].max().item(),
            #                   "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
            #                   "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
            #                   "success_all": self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
            #                   "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
            #                   "success_query_all": self.success_query_all.detach().cpu().numpy().astype(
            #                       np.int32).tolist(),
            #                   "distortion": self.distortion_all,
            #                   "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
            #                   "args": vars(args)}
            # with open(result_dump_path, "w") as result_file_obj:
            #     json.dump(meta_info_dict, result_file_obj, sort_keys=True)


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



def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'QEBA_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'QEBA-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/QEBA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'load_random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_discretize', action="store_true")
    parser.add_argument('--atk_level', type=int, default=999)
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument("--num_iterations",type=int,default=64)
    parser.add_argument('--stepsize_search', type=str, choices=['geometric_progression', 'grid_search'],default='geometric_progression')
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--max_queries',type=int, default=10000)
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--max_num_evals', type=int,default=100)
    parser.add_argument('--pgen',type=str,choices=['naive',"resize","DCT9408","DCT192"],required=True)
    args = parser.parse_args()
    assert args.batch_size == 1, "HSJA only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    # if args.targeted:
    #     if args.dataset == "ImageNet":
    #         args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_pgen_{}_defense_{}.log'.format(args.pgen,args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_pgen_{}.log'.format(args.pgen))
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_pgen_{}_defense_{}_{}.log'.format(args.pgen,args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_pgen_{}_{}.log'.format(args.pgen,args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.all_archs:
        archs = args.all_archs
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    PGEN = args.pgen
    p_gen = load_pgen(args.dataset, PGEN, args)
    if args.dataset.startswith("CIFAR"):
        if PGEN == 'naive':
            ITER = 150
            maxN = 30
            initN = 30
        elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
            ITER = 150
            maxN = 30
            initN = 30
        elif PGEN.startswith('PCA'):
            ITER = 150
            maxN = 30
            initN = 30
        else:
            raise NotImplementedError()
    elif args.dataset == 'ImageNet' or args.dataset == 'CelebA':
        if PGEN == 'naive':
            ITER = 100
            maxN = 100
            initN = 100
        elif PGEN.startswith('PCA'):
            ITER = 100
            maxN = 100
            initN = 100
        elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
            ITER = 100
            maxN = 100
            initN = 100
        elif PGEN == 'NNGen':
            ITER = 500
            maxN = 30
            initN = 30
    maxN = 10000  # FIXME 原来的梯度估计花费的上限太小了，和我的HSJA等比较不公平!
    initN = 100
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_pgen_{}_result.json".format(arch, args.defense_model,args.pgen)
        else:
            save_result_path = args.exp_dir + "/{}_pgen_{}_result.json".format(arch,args.pgen)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = QEBA(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                        args.norm, args.epsilon, iterations=ITER, initial_num_evals=initN, max_num_evals=maxN,
                       internal_dtype=torch.float32,rv_generator=p_gen, atk_level=args.atk_level, mask=None,
                        gamma=args.gamma, batch_size=100, stepsize_search = args.stepsize_search,
                        log_every_n_steps=1, suffix=PGEN, verbose=False, maximum_queries=args.max_queries)
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
