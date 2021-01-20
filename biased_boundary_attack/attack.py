import argparse
import sys
import os
sys.path.append(os.getcwd())
import json
import random

from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import numpy as np
import os
import torch
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F
import glog as log

from biased_boundary_attack.utils.sampling.sampling_provider import SamplingProvider
from biased_boundary_attack.utils.util import sample_hypersphere
from config import CLASS_NUM, IMAGE_DATA_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
import os.path as osp

class BiasedBoundaryAttack(object):
    """
    Guessing Smart paper
     Like BoundaryAttack, but uses biased sampling from various sources.
     This implementation is optimized for speed: it can query the model and, while waiting, already prepare the next perturbation candidate.
     Ideally, there is zero overhead over the prediction speed of the model under attack.

     However, we do NOT run predictions in parallel (as the Foolbox BoundaryAttack does).
     This attack is completely sequential to keep the number of queries minimal.

     Activate various biases in bench_settings.py.
    """

    def __init__(self, model, substitute_model,  dataset, norm, epsilon,
                 batch_size, targeted, clip_min, clip_max, maximum_queries,
                 USE_PERLIN_BIAS, USE_MASK_BIAS, USE_SURROGATE_BIAS):
        """
        Creates an instance that can be reused when attacking multiple images.
        :param model: The model to attack.
        :param sample_gen: Random sample generator, which is utils/sampling/sampling_provider.py
        :param substitute_model: A differentiable surrogate model for gradients. If None, then the surrogate bias will not be used.
        """

        self.model = model
        # A substitute model that provides batched gradients.
        self.substitute_model = substitute_model
        #self.dm_main = dm_main.to_range_01()          # Images are normed to 0/1 inside of run_attack()
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted #  True if targeted attack.
        self.USE_PERLIN_BIAS = USE_PERLIN_BIAS
        self.USE_MASK_BIAS = USE_MASK_BIAS
        self.USE_SURROGATE_BIAS = USE_SURROGATE_BIAS
        # We use ThreadPools to calculate candidates and surrogate gradients while we're waiting for the model's next prediction.
        self.pg_thread_pool = ThreadPoolExecutor(max_workers=1)
        self.candidate_thread_pool = ThreadPoolExecutor(max_workers=1)
        # img_shape =(self.model.input_size[1],self.model.input_size[2],self.model.input_size[0])
        img_shape = self.model.input_size
        self.sample_gen = SamplingProvider(shape=img_shape, n_threads=3, queue_lengths=80)
        self.epsilon = epsilon
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
        self.distortion_best_all = torch.zeros_like(self.query_all)

    def __enter__(self):
        self.pg_thread_pool.__enter__()
        self.candidate_thread_pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Will block until the futures are calculated. Thankfully they're not very complicated.
        self.pg_thread_pool.__exit__(exc_type, exc_value, traceback)
        self.candidate_thread_pool.__exit__(exc_type, exc_value, traceback)
        log.info("BiasedBoundaryAttack: all threads stopped.")

    def line_search_to_boundary(self, x_orig, x_start, label, is_targeted):
        """
        Binary search along a line between start and original image in order to find the decision boundary.
        :param x_orig: The original image to attack.
        :param x_start: The starting image (which fulfills the adversarial criterion)
        :param is_targeted: true if the attack is targeted.
        :param label: the target label if targeted, or the correct label if untargeted.
        :return: A point next to the decision boundary (but still adversarial)
        """

        eps = 0.5  # Stop when decision boundary is closer than this (in L2 distance)
        i = 0

        x1 = x_start.float()
        x2 = x_orig.float()
        diff = x2 - x1
        while torch.linalg.norm(diff) > eps:
            i += 1

            x_candidate = x1 + 0.5 * diff

            self.query += 1
            if (self.model(x_candidate).max(1)[1].item() == label) == is_targeted:
                x1 = x_candidate
            else:
                x2 = x_candidate

            diff = x2 - x1

        log.info("Found decision boundary after {} queries.".format(i))
        return x1


    def find_closest_img(self, X_orig, X_targets, label, is_targeted):
        """
        From a list of potential starting images, finds the closest to the original.
        Before returning, this method makes sure that the image fulfills the adversarial condition (is actually classified as the target label).
        :param X_orig: The original image to attack.
        :param X_targets: List of images that fulfill the adversarial criterion (i.e. target class in the targeted case)
        :param is_targeted: true if the attack is targeted.
        :param label: the target label if targeted, or the correct label if untargeted.
        :return: the closest image (in L2 distance) to the original that also fulfills the adversarial condition.
        """

        dists = torch.empty(X_targets.size(0), dtype=torch.float32).to(X_targets.device)
        for i in range(X_targets.size(0)):
            d_l2 = torch.linalg.norm(X_targets[i] - X_orig)
            dists[i] = d_l2

        indices = torch.argsort(dists)
        for index in indices:
            X_target = X_targets[index]
            if X_target.dim() == 3:
                X_target = X_target.unsqueeze(0)
            pred_clsid = self.model(X_target).max(1)[1].item()
            if (pred_clsid == label) == is_targeted:
                log.info("Found an image of the target class, d_l2={:.3f}.".format(dists[index]))
                return X_target

            log.info("Image of target class is wrongly classified by model, skipping.")

        return X_targets[random.randint(0,X_targets.size(0)-1)]

    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):

        dist = torch.norm((perturbed - images).view(1, -1), self.ord, 1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                inside_batch_index].item()
        return dist

    def attack(self, image_index, image,image_start,label, n_max_per_batch=50,
               source_step=1e-2, spherical_step=1e-2, mask=None, recalc_mask_every=None):
        """
        Runs the Biased Boundary Attack against a single image.
        .

        :param image: The original (clean) image to perturb.
        :param image_start: The starting point (must be of target class).
        :param label: The target label (if targeted), or the original label (if untargeted).
        :param n_max_per_batch: How many samples are drawn per "batch". Samples are processed serially (the challenge doesn't allow
                                batching), but for each sample, the attack dynamically adjusts hyperparams based on the success of
                                previous samples. This "batch" size is the max number of samples after which hyperparams are reset, and
                                a new "batch" is started. See generate_candidate().
        :param source_step: source step hyperparameter (see Boundary Attack)
        :param spherical_step: orthogonal step hyperparameter (see Boundary Attack)
        :param mask: Optional. If not none, a predefined mask (expert knowledge?) can be defined that will be applied to the perturbations.
        :param recalc_mask_every: If not none, automatically calculates a mask from the current image difference.
                                  Will recalculate this mask every (n) steps.
        :return: The best adversarial example so far.
        """
        success_stop_queries = torch.zeros(1)
        batch_image_positions = np.arange(image_index , min(image_index + 1, self.total_images)).tolist()

        n_calls_left_fn = lambda q: self.maximum_queries - int(q[0].item())  # The attack terminates when n_calls_left_fn() returns 0
        if image.dim() == 4:
            image = torch.squeeze(image,0)
        if image_start is not None and image_start.dim() == 4:
            image_start = torch.squeeze(image_start,0)
        assert image.dim() == 3
        assert image_start.dim() == 3
        if mask is not None:
            assert mask.size() == image.size()
            assert torch.sum((mask < 0).int()) == 0 and 1. - torch.max(mask).item() < 1e-4, "Mask must be scaled to [0,1]. At least one value must be 1."
        else:
            mask = torch.ones_like(image).type(torch.float32)

        # time_start = timeit.default_timer()
        pg_future = None
        try:
            label_current, dist_best = self._eval_sample(image_start, image)
            if (label_current == label) != self.targeted:
                log.info("WARN: Starting point is not a valid adversarial example! Continuing for now.")

            X_adv_best = image_start.clone()

            last_mask_recalc_calls = n_calls_left_fn(self.query)

            # Abort if we're running out of queries
            while n_calls_left_fn(self.query) > 3:
                # Mask Bias: recalculate mask from current diff (hopefully this reduces the search space)
                if recalc_mask_every is not None and last_mask_recalc_calls - n_calls_left_fn(self.query) >= recalc_mask_every:
                    new_mask = torch.abs(X_adv_best - image)
                    new_mask = new_mask / torch.max(new_mask)             # scale to [0,1]
                    new_mask = torch.sqrt(new_mask)                     # weaken the effect a bit.
                    log.info("{}-th image, recalculated mask. Weighted dimensionality of search space is now {:.0f} (diff: {:.2%}). ".format(
                           image_index+1, torch.sum(new_mask).item(), 1. - torch.sum(new_mask).item() / torch.sum(mask).item()))
                    mask = new_mask
                    last_mask_recalc_calls = n_calls_left_fn(self.query)

                # Draw n candidates at the current position (before resetting hyperparams or before reaching the limit)
                n_candidates = min(n_max_per_batch, n_calls_left_fn(self.query))

                # Calculate the projected adversarial surrogate gradient at the current position.
                #  Putting this into a ThreadPoolExecutor. While this is processing, we can already start drawing the first sample.
                # Also cancel any pending requests from previous steps.
                if pg_future is not None:
                    pg_future.cancel()
                pg_future = self.pg_thread_pool.submit(self.get_projected_gradients, **{
                    "x_current": X_adv_best,
                    "x_orig": image,
                    "label": label,
                    "is_targeted": self.targeted})

                # Also do candidate generation with a ThreadPoolExecutor.
                # Queue the first candidate.
                candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                    "i": 0,
                    "n": n_candidates,
                    "x_orig": image,
                    "x_current": X_adv_best,
                    "mask": mask,
                    "source_step": source_step,
                    "spherical_step": spherical_step,
                    "pg_future": pg_future})
                for i in range(n_candidates):
                    # Get candidate and queue the next one.
                    candidate, stats = candidate_future.result()
                    if i < n_candidates - 1:
                        candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                            "i": i+1,
                            "n": n_candidates,
                            "x_orig": image,
                            "x_current": X_adv_best,
                            "mask": mask,
                            "source_step": source_step,
                            "spherical_step": spherical_step,
                            "pg_future": pg_future})

                    candidate_label, dist = self._eval_sample(candidate, image)
                    if self.targeted:
                        if candidate_label == label:  # 只有攻击错误成功时，才更新dist_best和X_adv_best
                            if dist.item() < dist_best.item():
                                X_adv_best = candidate
                                dist_best = dist
                                distortion = self.count_stop_query_and_distortion(image, X_adv_best, self.query,
                                                                                  success_stop_queries,
                                                                                  batch_image_positions)
                                log.info(
                                    "{}-th image, distortion:{:.4f} query:{}".format(image_index + 1, distortion.item(),
                                                                                     self.query[0].item()))
                                # log.info("{}-th image,break this batch and candidates".format(image_index+1))
                                break  # Terminate this batch (don't try the other candidates) and advance.
                    else:
                        if candidate_label != label:
                            if dist.item() < dist_best.item():
                                X_adv_best = candidate
                                dist_best = dist
                                distortion = self.count_stop_query_and_distortion(image, X_adv_best, self.query,
                                                                                  success_stop_queries,
                                                                                  batch_image_positions)
                                log.info(
                                    "{}-th image, distortion:{:.4f} query:{}".format(image_index + 1, distortion.item(),
                                                                                     self.query[0].item()))
                                # log.info("{}-th image,break this batch and candidates".format(image_index + 1))
                                break  # Terminate this batch (don't try the other candidates) and advance.
                    if self.query[0].item() >= self.maximum_queries:
                        break

            success_stop_queries = torch.clamp(success_stop_queries,0,self.maximum_queries)
            return X_adv_best, self.query.clone(), success_stop_queries, dist_best,  (dist_best <= self.epsilon)
        finally:
            # Be safe and wait for the gradient future. We want to be sure that no BG worker is blocking the GPU before returning.
            if pg_future is not None:
                futures.wait([pg_future])

    def generate_candidate(self, i, n, x_orig, x_current, mask, source_step, spherical_step, pg_future):

        # This runs in a loop (while i<n) per "batch".
        # Whenever a candidate is successful, a new batch is started. Therefore, i is the number of previously unsuccessful samples.
        # Trying to use this in our favor, we progressively reduce step size for the next candidate.
        # When the batch is through, hyperparameters are reset for the next batch.

        # Scale both spherical and source step with i.
        scale = (1. - i/n) + 0.3
        c_source_step = source_step * scale
        c_spherical_step = spherical_step * scale

        # Get the adversarial projected gradient from the (other) BG worker.
        pg_factor = 0.3
        pgs = pg_future.result()
        pgs = pgs if i % 2 == 0 else None           # Only use gradient bias on every 2nd iteration, but always try it at first..

        if self.USE_PERLIN_BIAS:
            sampling_fn = self.sample_gen.get_perlin
        else:
            sampling_fn = self.sample_gen.get_normal

        candidate, stats = self.generate_boundary_sample(
            X_orig=x_orig, X_adv_current=x_current, mask=mask, source_step=c_source_step, spherical_step=c_spherical_step,
            sampling_fn=sampling_fn, pgs_current=pgs, pg_factor=pg_factor)

        stats["i_sample"] = int(i)
        stats["mask_sum"] = float(torch.sum(mask).item())
        return candidate, stats

    def generate_boundary_sample(self, X_orig, X_adv_current, mask, source_step, spherical_step, sampling_fn,
                                 pgs_current=None, pg_factor=0.5):
        # Adapted from FoolBox BoundaryAttack.

        unnormalized_source_direction = X_orig.float() - X_adv_current.float()
        source_norm = torch.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        # Get perturbation from provided distribution
        sampling_dir, stats = sampling_fn(return_stats=True)
        sampling_dir = torch.from_numpy(sampling_dir).to(source_direction.device)  # 因为sampling_fn都是基于numpy写的
        # ===========================================================
        # calculate candidate on sphere
        # ===========================================================
        dot = torch.vdot(sampling_dir.view(-1), source_direction.view(-1))
        sampling_dir -= dot * source_direction                                      # Project orthogonal to source direction
        sampling_dir *= mask                                                        # Apply regional mask
        sampling_dir /= torch.linalg.norm(sampling_dir)                                 # Norming increases magnitude of masked regions
        # If available: Bias the spherical dirs in direction of the adversarial gradient, which is projected onto the sphere
        if pgs_current is not None:
            # We have a bunch of gradients that we can try. Randomly select one.
            # NOTE: we found this to perform better than simply averaging the gradients.
            pg_current = pgs_current[np.random.randint(0, len(pgs_current))]
            pg_current *= mask
            pg_current /= torch.linalg.norm(pg_current)

            sampling_dir = (1. - pg_factor) * sampling_dir + pg_factor * pg_current
            sampling_dir /= torch.linalg.norm(sampling_dir)

        sampling_dir *= spherical_step * source_norm                                # Norm to length stepsize*(dist from src)

        D = 1 / np.sqrt(spherical_step ** 2 + 1)
        direction = sampling_dir - unnormalized_source_direction
        spherical_candidate = X_orig + D * direction

        torch.clamp(spherical_candidate, min=self.clip_min, max=self.clip_max, out=spherical_candidate)

        # ===========================================================
        # step towards source
        # ===========================================================
        new_source_direction = X_orig - spherical_candidate

        new_source_direction_norm = torch.linalg.norm(new_source_direction)
        new_source_direction /= new_source_direction_norm
        spherical_candidate = X_orig - source_norm * new_source_direction           # Snap sph.c. onto sphere

        # From there, take a step towards the target.
        candidate = spherical_candidate + (source_step * source_norm) * new_source_direction

        torch.clamp(candidate, min=self.clip_min, max=self.clip_max, out=candidate)
        return candidate.float(), stats


    def get_gradient(self, model, x, labels):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x.requires_grad_()
        assert x.dim() == 4, x.dim()
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(loss, x, retain_graph=False)[0].detach()
        return grads

    def get_projected_gradients(self, x_current, x_orig, label, is_targeted):
        # Idea is: we have a direction (spherical candidate) in which we want to sample.
        # We know that the gradient of a substitute model, projected onto the sphere, usually points to an adversarial region.
        # Even if we are already adversarial, it should point "deeper" into that region.
        # If we sample in that direction, we should move toward the center of the adversarial cone.
        # Here, we simply project the gradient onto the same hyperplane as the spherical samples.
        #
        # Instead of a single projected gradient, this method returns an entire batch of them:
        # - Surrogate gradients are unreliable, so we sample them in a region around the current position.
        # - This gives us a similar benefit as observed in "PGD with random restarts".

        if self.substitute_model is None:
            return None

        source_direction = x_orig - x_current
        source_norm = torch.linalg.norm(source_direction).item() # float
        source_direction = source_direction / source_norm

        # Take a tiny step towards the source before calculating the gradient. This marginally improves our results.
        step_inside = 1e-2 * source_norm
        x_inside = x_current + step_inside * source_direction

        # Perturb the current position before calc'ing gradient
        n_samples = 4
        radius_max = 1e-2 * source_norm
        x_perturb = sample_hypersphere(n_samples=n_samples, sample_shape=list(x_orig.size()), radius=1, sample_gen=self.sample_gen)
        x_perturb *= np.random.uniform(0., radius_max)  # float
        x_perturb = x_perturb.to(x_inside.device)

        x_inside_batch = x_inside + x_perturb
        gradient_batch = torch.empty_like(x_inside_batch).float()
        gradients = self.get_gradient(self.substitute_model, x_inside_batch, torch.tensor([label] * n_samples).long().cuda())
        if is_targeted:
            gradients = -gradients

        for i in range(n_samples):
            # Project the gradients.
            dot = torch.vdot(gradients[i].view(-1), source_direction.view(-1))
            projected_gradient = gradients[i] - dot * source_direction          # Project orthogonal to source direction
            projected_gradient /= torch.linalg.norm(projected_gradient)            # Norm to length 1
            gradient_batch[i] = projected_gradient

        return gradient_batch

    def _eval_sample(self, x, x_orig=None):
        # Round, then get label and distance.
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x_orig.dim() ==3:
            x_orig = x_orig.unsqueeze(0)

        assert x.dim() == 4, "x dim={}".format(x.dim())
        assert x_orig.dim() ==4,"x_orig dim={}".format(x_orig.dim())
        preds = self.model(x)
        self.query += x.size(0)
        label = torch.argmax(preds,dim=1).item()

        if x_orig is None:
            return label
        else:
            dist = torch.norm((x - x_orig).view(1, -1), p=self.ord)
            return label, dist

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

    def attack_all_images(self, args, arch_name, result_dump_path):

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            images = images.cuda()
            with torch.no_grad():
                logit = self.model(images)
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
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = self.get_image_of_target_class(self.dataset_name,target_labels, self.model)

                label = target_labels[0].item()
            else:
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                              size=true_labels.size()).long()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long()
                    invalid_target_index = target_labels.eq(true_labels)
                target_images = self.get_image_of_target_class(self.dataset_name, target_labels, self.model)
                label = true_labels[0].item()
            target_images = target_images.cuda()
            target_images = self.find_closest_img(images, target_images, label, self.targeted)
            self.query = torch.zeros(1)
            target_images = self.line_search_to_boundary(x_orig=images, x_start=target_images, label=label,
                                              is_targeted=self.targeted)
            adv_images, query, success_query, distortion_best, success_epsilon = self.attack(batch_index, images, target_images,
                                                                                    label, source_step=2e-3,spherical_step=5e-2,mask=None,
                                                                                    recalc_mask_every=(1 if self.USE_MASK_BIAS else None))
            distortion_best = distortion_best.detach().cpu()
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
            success = (1 - not_done.detach().cpu()) * success_epsilon.float().detach().cpu()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_best"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()
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
                          "avg_distortion_best": self.distortion_best_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'biased_boundary_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'biased_boundary_attack-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/BBA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--surrogate_arch',type=str,default="inceptionresnetv2")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--use_perlin_bias',action="store_true")
    parser.add_argument('--use_mask_bias', action="store_true")
    parser.add_argument('--use_surrogate_bias',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str,default="")
    parser.add_argument('--max_queries',type=int, default=10000)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    assert args.batch_size == 1, "BBA only supports mini-batch size equals 1!"
    assert args.norm == "l2", "Please modify line_search_to_boundary function in Linf norm attack"
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted and args.dataset == "ImageNet":
            args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type,
                                             args))  # 随机产生一个目录用于实验
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
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
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
    substitude_model = None
    if args.use_surrogate_bias:
        substitude_model = StandardModel(args.dataset,args.surrogate_arch,no_grad=False,load_pretrained=True)
        substitude_model.cuda()
        substitude_model.eval()

    for arch in archs:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model, norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = BiasedBoundaryAttack(model, substitude_model, args.dataset, args.norm,args.epsilon,args.batch_size,
                                        args.targeted, 0.0, 1.0, args.max_queries, args.use_perlin_bias,
                                        args.use_mask_bias, args.use_surrogate_bias)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
