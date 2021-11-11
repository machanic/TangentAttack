import argparse

import os
import random
import sys
sys.path.append(os.getcwd())
from collections import defaultdict, OrderedDict

import json
from types import SimpleNamespace
import os.path as osp
import torch
import numpy as np
import glog as log
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from dataset.target_class_dataset import ImageNetDataset,CIFAR100Dataset,CIFAR10Dataset,TinyImageNetDataset
from tangent_attack_hemisphere.scipy_find_tangent_point import solve_tangent_point
from tangent_attack_hemisphere.tangent_point_analytical_solution import TangentFinder

class TangentAttack(object):
    def __init__(self, model, dataset,  clip_min, clip_max, height, width, channels, norm, epsilon, iterations=40, gamma=1.0,
                 stepsize_search='geometric_progression',
                 max_num_evals=1e4, init_num_evals=100, maximum_queries=10000, batch_size=1,
                 verify_tangent_point=False,best_radius=False):
        """
        :param clip_min: lower bound of the image.
        :param clip_max: upper bound of the image.
        :param norm: choose between [l2, linf].
        :param iterations: number of iterations.
        :param gamma: used to set binary search threshold theta. The binary search
                     threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for linf attack.
        :param stepsize_search: choose between 'geometric_progression', 'grid_search'.
        :param max_num_evals: maximum number of evaluations for estimating gradient.
        :param init_num_evals: initial number of evaluations for estimating gradient.
        """
        self.model = model
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dim = height * width * channels
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (channels, height, width)
        if self.norm == "l2":
            self.theta = gamma / (np.sqrt(self.dim) * self.dim)
        else:
            self.theta = gamma / (self.dim ** 2)
        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.num_iterations = iterations
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.verify_tangent_point = verify_tangent_point

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.best_radius = best_radius

    def get_image_of_target_class(self,dataset_name, target_labels):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name=="CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "TinyImageNet":
                dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and self.model.input_size[-1] != 299:
                image = F.interpolate(image,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logits = self.model(image.cuda())
            while logits.max(1)[1].item() != label.item():
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and self.model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
                with torch.no_grad():
                    logits = self.model(image.cuda())
            assert true_label == label.item()
            images.append(torch.squeeze(image))
        return torch.stack(images) # B,C,H,W

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1].detach().cpu() != true_labels
        else:
            return logits.max(1)[1].detach().cpu() == target_labels


    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise[None], true_labels, target_labels)[0].item()
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

                    initialization = self.get_image_of_target_class(self.dataset_name,target_labels).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended[None], true_labels, target_labels)[0].item()
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization, num_eval

    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.min(torch.max(image, clip_min), clip_max)


    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [alphas.size(0)] + [1] * len(self.shape)
        alphas = alphas.view(*alphas_shape)
        if self.norm == 'l2':
            return (1 - alphas) * original_image + alphas * perturbed_images
        elif self.norm == 'linf':
            out_images = self.clip_image(perturbed_images, original_image - alphas, original_image + alphas)
            return out_images

    def binary_search_batch(self, original_image, perturbed_images, true_labels, target_labels):
        num_evals = 0
        # Compute distance between each of perturbed image and original image.
        dists_post_update = torch.tensor([
            self.compute_distance(
                original_image,
                perturbed_image,
                self.norm
            ) for perturbed_image in perturbed_images])
        # Choose upper thresholds in binary searchs based on constraint.
        if self.norm == "linf":
            highs = dists_post_update # Stopping criteria.
            thresholds = torch.clamp_max(dists_post_update * self.theta, max=self.theta)
        else:
            highs = torch.ones(perturbed_images.size(0))
            thresholds = self.theta
        lows = torch.zeros(perturbed_images.size(0))
        # Call recursive function.

        while torch.max((highs - lows) / thresholds).item() > 1:
            # log.info("max in binary search func: {}, highs:{}, lows:{}, highs-lows: {} , threshold {}, (highs - lows) / thresholds: {}".format(torch.max((highs - lows) / thresholds).item(),highs, lows, highs-lows, thresholds, (highs - lows) / thresholds))
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids)
            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_images, true_labels, target_labels)
            num_evals += mid_images.size(0)
            decisions = decisions.int()
            # 攻击成功时候high用mid，攻击失败的时候low用mid
            lows = torch.where(decisions == 0, mids, lows)  # lows:攻击失败的用mids，攻击成功的用low
            highs = torch.where(decisions == 1, mids, highs)  # highs: 攻击成功的用mids，攻击失败的用high, 不理解的可以去看论文Algorithm 1
            # log.info("decision: {} low: {}, high: {}".format(decisions.detach().cpu().numpy(),lows.detach().cpu().numpy(), highs.detach().cpu().numpy()))
        out_images = self.project(original_image, perturbed_images, highs)  # high表示classification boundary偏攻击成功一点的线
        # Compute distance of the output image to select the best choice. (only used when stepsize_search is grid_search.)
        dists = torch.tensor([
            self.compute_distance(
                original_image,
                out_image,
                self.norm
            ) for out_image in out_images])
        idx = torch.argmin(dists)
        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist, num_evals

    def select_delta(self, cur_iter, dist_post_update):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.

        """
        if cur_iter == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.norm == 'l2':
                delta = np.sqrt(self.dim) * self.theta * dist_post_update
            elif self.norm == 'linf':
                delta = self.dim * self.theta * dist_post_update
        return delta

    def approximate_gradient(self, sample, true_labels, target_labels, num_evals, delta):
        clip_max, clip_min = self.clip_max, self.clip_min

        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)
        if self.norm == 'l2':
            rv = torch.randn(*noise_shape)
        elif self.norm == 'linf':
            rv = torch.from_numpy(np.random.uniform(low=-1, high=1, size=noise_shape)).float()
            # rv = torch.FloatTensor(*noise_shape).uniform_(-1, 1)
        rv = rv / torch.sqrt(torch.sum(torch.mul(rv,rv), dim=(1,2,3),keepdim=True))
        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        # if self.dataset_name=="ImageNet" and perturbed.size(0) >= 4:  # FIXME save GPU memory
        #     decisions_1 = self.decision_function(perturbed[:perturbed.size(0)//4], true_labels, target_labels)
        #     decisions_2 = self.decision_function(perturbed[perturbed.size(0) // 4: perturbed.size(0) * 2 // 4], true_labels, target_labels)
        #     decisions_3 = self.decision_function(perturbed[perturbed.size(0)*2 // 4:perturbed.size(0) * 3 // 4],
        #                                          true_labels, target_labels)
        #     decisions_4 = self.decision_function(perturbed[perturbed.size(0) * 3 // 4:],
        #                                          true_labels, target_labels)
        #     decisions = torch.cat([decisions_1, decisions_2,decisions_3,decisions_4],0)
        # else:
        decisions = self.decision_function(perturbed, true_labels, target_labels)
        decision_shape = [decisions.size(0)] + [1] * len(self.shape)
        fval = 2 * decisions.float().view(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval).item() == 1.0:  # label changes.
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval).item() == -1.0:  # label not change.
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf,p=2)

        return gradf

    def geometric_progression_for_HSJA(self, x, true_labels, target_labels, update, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        epsilon = dist.item() / np.sqrt(cur_iter)
        num_evals = np.zeros(1)
        def phi(epsilon, num_evals):
            new = x + epsilon * update
            success = self.decision_function(new[None], true_labels,target_labels)
            num_evals += 1
            return bool(success[0].item())

        while not phi(epsilon, num_evals):  # 只要没有成功，就缩小epsilon
            epsilon /= 2.0
        perturbed = torch.clamp(x + epsilon * update, self.clip_min, self.clip_max)
        return perturbed


    def geometric_progression_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels,
                                                 dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        radius = dist.item() / np.sqrt(cur_iter)
        num_evals = 0
        while True:
            # x_projection = calculate_projection_of_x_original(x_original.view(-1),x_boundary.view(-1),normal_vector.view(-1))
            # if torch.norm(x_projection.view(-1) - x_original.view(-1),p=self.ord).item() <= radius:
            #     log.info("projection point lies inside ball! reduce radius from {:.3f} to {:.3f}".format(radius, radius/2.0))
            #     radius /= 2.0
            #     continue
            # else:
            tangent_finder = TangentFinder(x_original.view(-1), x_boundary.view(-1), radius, normal_vector.view(-1),
                                           norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            if self.verify_tangent_point:
                log.info("verifying tagent point")
                another_tangent_point = solve_tangent_point(x_original.view(-1).detach().cpu().numpy(), x_boundary.view(-1).detach().cpu().numpy(),
                                    normal_vector.view(-1).detach().cpu().numpy(), radius, clip_min=self.clip_min,clip_max=self.clip_max)
                if isinstance(another_tangent_point, np.ndarray):
                    another_tangent_point = torch.from_numpy(another_tangent_point).type_as(tangent_point).to(tangent_point.device)
                difference = tangent_point - another_tangent_point
                log.info("Difference max: {:.4f} mean: {:.4f} sum: {:.4f} L2 norm: {:.4f}".format(difference.max().item(),
                                                                                                  difference.mean().item(),
                                                                                                  difference.sum().item(),
                                                                                                  torch.norm(difference)))
            tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)
            num_evals += 1
            if bool(success[0].item()):
               break
            radius /= 2.0
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals


    def log_geometric_progression_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        radius = dist.item() / np.log2(cur_iter+1)
        num_evals = 0
        while True:

            tangent_finder = TangentFinder(x_original.view(-1), x_boundary.view(-1), radius, normal_vector.view(-1), norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            if self.verify_tangent_point:
                log.info("verifying tagent point")
                another_tangent_point = solve_tangent_point(x_original.view(-1).detach().cpu().numpy(), x_boundary.view(-1).detach().cpu().numpy(),
                                    normal_vector.view(-1).detach().cpu().numpy(), radius, clip_min=self.clip_min,clip_max=self.clip_max)
                if isinstance(another_tangent_point, np.ndarray):
                    another_tangent_point = torch.from_numpy(another_tangent_point).type_as(tangent_point).to(tangent_point.device)
                difference = tangent_point - another_tangent_point
                log.info("Difference max: {:.4f} mean: {:.4f} sum: {:.4f} L2 norm: {:.4f}".format(difference.max().item(),
                                                                                                  difference.mean().item(),
                                                                                                  difference.sum().item(),
                                                                                                  torch.norm(difference)))
            tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)
            num_evals += 1
            if bool(success[0].item()):
               break
            radius /= 2.0
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals


    def fixed_radius_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels, radius):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """

        tangent_finder = TangentFinder(x_original.view(-1), x_boundary.view(-1), radius, normal_vector.view(-1), norm="l2")
        tangent_point = tangent_finder.compute_tangent_point()
        tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        success = self.decision_function(tangent_point[None], true_labels, target_labels)
        return tangent_point, bool(success[0].item())

    def binary_search_for_radius_and_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels,
                                                   dist):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        num_evals = 0
        low = 0
        high = dist.item()
        while high - low > 0.1:
            mid = (high + low) / 2.0
            tangent_finder = TangentFinder(x_original.view(-1), x_boundary.view(-1), mid, normal_vector.view(-1),
                                           norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)
            success = self.decision_function(tangent_point[None], true_labels, target_labels)[0].item()
            num_evals += 1
            if success:
                high = mid
            else:
                low = mid
        tangent_finder = TangentFinder(x_original.view(-1), x_boundary.view(-1), high, normal_vector.view(-1),
                                       norm="l2")
        tangent_point = tangent_finder.compute_tangent_point()
        tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point, num_evals

    def compute_distance(self, x_ori, x_pert, norm='l2'):
        # Compute the distance between two images.
        if norm == 'l2':
            return torch.norm(x_ori - x_pert,p=2).item()
        elif norm == 'linf':
            return torch.max(torch.abs(x_ori - x_pert)).item()

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()
        assert images.size(0) == 1
        batch_size = images.size(0)
        images = images.squeeze()
        if target_images is not None:
            target_images = target_images.squeeze()
        # Initialize.
        perturbed, num_eval = self.initialize(images, target_images, true_labels, target_labels)
        # log.info("after initialize")
        query += num_eval
        dist =  torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        # Project the initialization to the boundary.
        # log.info("before first binary_search_batch")
        perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbed[None], true_labels,target_labels)
        # log.info("after first binary_search_batch")
        dist =  torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        query += num_eval
        success_stop_queries[working_ind] = query[working_ind]

        cur_iter = 0
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()
        fixed_radius = dist.item() / np.sqrt(self.num_iterations)
        # init variables
        for j in range(self.num_iterations):
            cur_iter += 1
            # Choose delta.
            delta = self.select_delta(cur_iter, dist_post_update)
            # Choose number of evaluations.
            num_evals = int(self.init_num_evals * np.sqrt(j+1))
            num_evals = int(min([num_evals, self.max_num_evals]))
            # approximate gradient
            gradf = self.approximate_gradient(perturbed, true_labels, target_labels, num_evals, delta)
            if self.norm == "linf":
                gradf = torch.sign(gradf)

            query += num_evals
            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                # if not args.ablation_study: FIXME
                perturbed_Tagent, num_evals = self.geometric_progression_for_tangent_point(images, perturbed, gradf,
                                                                                       true_labels, target_labels,
                                                                                       dist, cur_iter)
                # else:  FIXME
                #     if args.fixed_radius:
                #         perturbed_Tagent, success = self.fixed_radius_for_tangent_point(images, perturbed, gradf, true_labels,
                #                                                                                    target_labels, fixed_radius)
                #         num_evals = torch.zeros_like(query)
                #     elif args.binary_search_radius:
                #         perturbed_Tagent, num_evals = self.binary_search_for_radius_and_tangent_point(images, perturbed, gradf,
                #                                                                                       true_labels, target_labels, dist)
                #     elif args.log2_radius:
                #         perturbed_Tagent, num_evals = self.log_geometric_progression_for_tangent_point(images, perturbed, gradf,
                #                                                                            true_labels, target_labels,
                #                                                                            dist, cur_iter)

                query += num_evals
                # perturbed_HSJA = self.geometric_progression_for_HSJA(perturbed, true_labels, target_labels, gradf, dist, cur_iter)
                # dist_tangent = torch.norm((perturbed_Tagent - images).view(batch_size, -1), self.ord, 1).item()
                # dist_HSJA = torch.norm((perturbed_HSJA - images).view(batch_size, -1), self.ord, 1).item()
                # log.info("dist of tangent: {:.4f}, dist of HSJA:{:.4f}, tangent < HSJA: {}".format(dist_tangent, dist_HSJA, dist_tangent < dist_HSJA))
                perturbed = perturbed_Tagent
                # Binary search to return to the boundary.
                # log.info("before geometric_progression binary_search_batch")
                perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbed[None], true_labels, target_labels)
                # log.info("after geometric_progression binary_search_batch")
                query += num_eval
                dist =  torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
                working_ind = torch.nonzero(dist > self.epsilon).view(-1)
                success_stop_queries[working_ind] = query[working_ind]
                for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                    self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                        inside_batch_index].item()
            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.

                update = gradf
                epsilons = torch.logspace(-4, 0, steps=20) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.view(epsilons_shape) * update
                perturbeds = torch.clamp(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = self.decision_function(perturbeds, true_labels, target_labels)
                query += perturbeds.size(0)
                for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                    self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                        inside_batch_index].item()
                if idx_perturbed.int().sum().item() > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update, num_eval = self.binary_search_batch(images, perturbeds[idx_perturbed], true_labels, target_labels)
                    query += num_eval
                    dist = torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
                    working_ind = torch.nonzero(dist > self.epsilon).view(-1)
                    success_stop_queries[working_ind] = query[working_ind]
                    for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                        self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                            inside_batch_index].item()
            if torch.sum(query >= self.maximum_queries).item() == true_labels.size(0):
                break
            # if not success:
            #     break
            # compute new distance.
            dist =  torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
            log.info('{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(batch_index+1, j + 1, self.norm, dist.item(), int(query[0].item())))
            if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                break
        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return perturbed, query, success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name,  result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0: # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index+1))
                continue
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

                target_images = self.get_image_of_target_class(self.dataset_name,target_labels)
            else:
                target_labels = None
                target_images = None

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            with torch.no_grad():
                if adv_images.dim() == 3:
                    adv_images = adv_images.unsqueeze(0)
                adv_logit = self.model(adv_images.cuda())
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
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item() if self.success_all.sum().item() > 0 else 0,
                          "median_query": self.success_query_all[self.success_all.bool()].median().item() if self.success_all.sum().item() > 0 else 0,
                          "max_query": self.success_query_all[self.success_all.bool()].max().item() if self.success_all.sum().item() > 0 else 0,
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
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.ablation_study:
        if args.attack_defense:
            dirname = 'tangent_attack_ablation_study_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'tangent_attack_ablation_study-{}-{}-{}'.format(dataset, norm, target_str)
        return dirname
    if args.init_num_eval_grad != 100:
        if args.attack_defense:
            dirname = 'tangent_attack@{}_on_defensive_model-{}-{}-{}'.format(args.init_num_eval_grad, dataset, norm, target_str)
        else:
            dirname = 'tangent_attack@{}-{}-{}-{}'.format(args.init_num_eval_grad, dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'tangent_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'tangent_attack-{}-{}-{}'.format(dataset, norm, target_str)

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
    parser.add_argument('--json-config', type=str, default='./configures/HSJA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--verify-tangent',action='store_true')

    # ablation study of the radius of the hemi-sphere
    parser.add_argument('--ablation-study',action='store_true')
    parser.add_argument('--binary-search-radius',action='store_true')
    parser.add_argument('--fixed-radius',action='store_true')
    parser.add_argument('--log2-radius', action='store_true')

    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random','load_random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument("--num_iterations",type=int,default=10000)
    parser.add_argument('--stepsize_search', type=str, choices=['geometric_progression', 'grid_search'],default='geometric_progression')
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str,default="")
    parser.add_argument('--max_queries',type=int, default=10000)
    parser.add_argument('--init_num_eval_grad',type=int,default=100)
    parser.add_argument('--gamma',type=float)

    args = parser.parse_args()
    assert args.batch_size == 1, "Tangent attack only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
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
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
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
            if args.defense_model == "adv_train_on_ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                               args.defense_norm,
                                                                               args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        elif args.ablation_study:
            if args.binary_search_radius:
                log_file_path = args.exp_dir + "/run_{}_binary_search_radius.log".format(args.arch)
            elif args.fixed_radius:
                log_file_path = args.exp_dir + "/run_{}_fixed_radius.log".format(args.arch)
            elif args.log2_radius:
                log_file_path = args.exp_dir + "/run_{}_log2_geometric_progression_radius.log".format(args.arch)
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))

    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        elif args.ablation_study:
            if args.binary_search_radius:
                save_result_path = args.exp_dir + "/{}_binary_search_radius_result.json".format(arch)
            elif args.fixed_radius:
                save_result_path = args.exp_dir + "/{}_fixed_radius_result.json".format(arch)
            elif args.log2_radius:
                save_result_path = args.exp_dir + "/{}_log2_geometric_progression_radius_result.json".format(arch)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = TangentAttack(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                     args.norm, args.epsilon, args.num_iterations, gamma=args.gamma, stepsize_search = args.stepsize_search,
                                     max_num_evals=1e4,
                                     init_num_evals=args.init_num_eval_grad, maximum_queries=args.max_queries,
                                     verify_tangent_point=args.verify_tangent)
        attacker.attack_all_images(args, arch,  save_result_path)
        model.cpu()
