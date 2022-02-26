import argparse
import os
import sys


sys.path.append(os.getcwd())
import json
import random
import sys
from collections import defaultdict, OrderedDict
from types import SimpleNamespace
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
import torch.nn as nn
import torchvision.datasets as dsets
import glog as log
import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import os.path as osp
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, PROJECT_PATH, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
from utils import get_label
from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
from numpy import linalg
import math
from GeoDA.utils import generate_2d_dct_basis
import time
from torch.nn import functional as F

class SubNoise(nn.Module):
    """given subspace x and the number of noises, generate sub noises"""

    # x is the subspace basis
    def __init__(self, num_noises, x, channels, image_height, image_width):
        self.num_noises = num_noises
        self.x = x
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        super(SubNoise, self).__init__()

    def forward(self):
        noise = torch.randn([self.x.shape[1], 3 * self.num_noises], dtype=torch.float32).cuda()
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([self.num_noises, self.channels, self.image_height, self.image_width])
        return r

class GeoDA(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, norm, epsilon,
                 search_space='sub', max_queries=10000, grad_estimator_batch_size=40, sub_dim=75, tol=0.0001,
                 sigma=0.0002, mu=0.6, batch_size=1):
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm = norm
        self.epsilon = epsilon
        self.channels = channels
        self.height = height
        self.width = width
        self.maximum_queries = max_queries
        self.sub_dim = sub_dim
        self.tol = tol
        self.mu = mu
        self.search_space = search_space
        self.verbose_control = True
        self.grad_estimator_batch_size = grad_estimator_batch_size # batch size for GeoDA
        self.sigma = sigma
        self.ord = np.inf if self.norm == "linf" else 2
        self.dataset = dataset
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

    def get_image_of_target_class(self,dataset_name, target_label, target_model):

        if dataset_name == "ImageNet":
            dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],target_label, "validation")
        elif dataset_name == "CIFAR-10":
            dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], target_label, "validation")
        elif dataset_name=="CIFAR-100":
            dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], target_label, "validation")

        index = np.random.randint(0, len(dataset))
        image, true_label = dataset[index]
        image = image.unsqueeze(0)
        if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
            image = F.interpolate(image,
                                   size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                   align_corners=False)
        with torch.no_grad():
            logits = target_model(image.cuda())
        while logits.max(1)[1].item() != target_label:
            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                   size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                   align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
        assert true_label == target_label
        return image # B,C,H,W



    def topk_3D(self, grad, k):
        grad_flatten = grad.cpu().numpy().reshape(-1)
        grad_flatten_torch = torch.tensor(grad_flatten)
        topk, indices = torch.topk(torch.abs(grad_flatten_torch), k)

        grad_k_flatten = torch.zeros([self.height * self.width * self.channels])

        for ind in indices:
            grad_k_flatten[ind] = grad_flatten[ind] + 0

        grad_k_flatten_np = grad_k_flatten.detach().cpu().numpy()

        grad_k_3D_np = np.reshape(grad_k_flatten_np, (self.channels, self.height, self.width))

        grad_3D_torch = torch.from_numpy(grad_k_3D_np)
        grad_3D_sign = torch.sign(grad_3D_torch)

        return grad_3D_sign

    def is_adversarial(self, given_image, true_label, target_label):
        predict_label = torch.argmax(self.model(given_image.cuda())).item()
        if target_label is None:
            return predict_label != true_label
        else:
            return predict_label == target_label

    def find_random_adversarial(self, image, true_label, target_label):
        num_calls = 0
        step = 0.02
        while True:
            pert = torch.randn([1, self.channels, self.height, self.width])
            perturbed = image + num_calls * step * pert
            perturbed = clip_image_values(perturbed, self.clip_min, self.clip_max)
            num_calls += 1
            if self.is_adversarial(perturbed, true_label, target_label):
                break
            if num_calls > 1000:
                log.info("Initialization failed! Use a misclassified image as `target_image")
                if target_label is None:
                    target_label = np.random.randint(0, CLASS_NUM[self.dataset])
                    while target_label == true_label:
                        target_label = np.random.randint(0, CLASS_NUM[self.dataset])

                perturbed = self.get_image_of_target_class(self.dataset, target_label, self.model).squeeze()
                return perturbed, 1

        return perturbed, num_calls


    def bin_search(self, x_0, x_random, true_label, target_label, tol):
        num_calls = 0
        adv = x_random
        cln = x_0

        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1
            if self.is_adversarial(mid, true_label, target_label):
                adv = mid
            else:
                cln = mid
            if torch.norm(adv - cln,p='fro').item() < tol:
                break
        return adv, num_calls

    def black_grad_batch(self, x_boundary, sub_basis_torch, q_max, sigma, batch_size, true_label, target_label):
        grad_tmp = []  # estimated gradients in each estimate_batch
        z = []  # sign of grad_tmp
        outs = []
        num_batchs = math.ceil(q_max / batch_size)
        last_batch = q_max - (num_batchs - 1) * batch_size
        EstNoise = SubNoise(batch_size, sub_basis_torch, self.channels, self.height, self.width).cuda()
        all_noises = []
        num_calls = 0
        for j in range(num_batchs):
            if j == num_batchs - 1:
                EstNoise_last = SubNoise(last_batch, sub_basis_torch, self.channels, self.height, self.width).cuda()
                current_batch = EstNoise_last()
                current_batch_np = current_batch.detach().cpu().numpy()
                # shape = [batch_size, num_noises, C, H, W]
                noisy_boundary = [x_boundary[0, :, :, :].cpu().numpy()] * last_batch + sigma * current_batch.detach().cpu().numpy()
            else:
                current_batch = EstNoise()
                current_batch_np = current_batch.detach().cpu().numpy()
                # shape = [batch_size, num_noises, C, H, W]
                noisy_boundary = [x_boundary[0, :, :, :].cpu().numpy()] * batch_size + sigma * current_batch.detach().cpu().numpy()

            all_noises.append(current_batch_np)
            # noisy_boundary shape = 40,3,224,224, current_batch shape = 40,3,224,224
            noisy_boundary_tensor = torch.tensor(noisy_boundary).cuda()  # [batch_size, num_noises, C, H, W]
            predict_labels = torch.argmax(self.model(noisy_boundary_tensor),1).cpu().numpy().astype(int)
            num_calls += noisy_boundary_tensor.size(0)
            outs.append(predict_labels)
        all_noise = np.concatenate(all_noises, axis=0)
        outs = np.concatenate(outs, axis=0)

        for i, predict_label in enumerate(outs):
            if target_label is not None:
                if predict_label != target_label:
                    z.append(1)
                    grad_tmp.append(all_noise[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-all_noise[i])
            else:
                if predict_label == true_label:  # predict == true label or predict != target class label
                    z.append(1)
                    grad_tmp.append(all_noise[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-all_noise[i])

        grad = -(1 / q_max) * sum(grad_tmp)

        grad_f = torch.tensor(grad)[None, :, :, :]

        return grad_f, sum(z), num_calls

    # FIXME not work, you can delete this code
    def geometric_progression_for_stepsize(self, x_adv, true_label, target_label, grad, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        epsilon = dist.item() / np.sqrt(cur_iter)
        num_evals = np.zeros(1)
        if self.norm == 'l1' or self.norm == 'l2':
            grads = grad
        if self.norm == 'linf':
            grads = torch.sign(grad) / torch.norm(grad)
        def phi(epsilon, num_evals):
            new = x_adv + epsilon * grads
            success = self.is_adversarial(new, true_label,target_label)
            num_evals += 1
            return success

        while not phi(epsilon, num_evals):  # 只要没有成功，就缩小epsilon
            epsilon /= 2.0
        perturbed = torch.clamp(x_adv + epsilon * grads, self.clip_min, self.clip_max)
        return perturbed, num_evals.item()


    def go_to_boundary(self, x_0, true_label, target_label, grad):
        epsilon = 5
        num_calls = 0

        if self.norm == 'l1' or self.norm == 'l2':
            grads = grad
        if self.norm == 'linf':
            grads = torch.sign(grad) / torch.norm(grad)
        while True:
            perturbed = x_0 + (num_calls * epsilon * grads[0])
            perturbed = clip_image_values(perturbed, self.clip_min, self.clip_max)
            num_calls += 1
            if self.is_adversarial(perturbed, true_label, target_label):
                break
            if num_calls > 100:
                log.info('failed ... ')
                break
        return perturbed, num_calls, epsilon * num_calls

    def calculate_distortion(self, x_adv, x_original, success_stop_queries, query, image_index):
        dist = torch.norm((x_adv - x_original).view(x_adv.size(0), -1), self.ord, 1).item()
        if dist > self.epsilon:
            success_stop_queries = query
        self.distortion_all[image_index][query] = dist
        return success_stop_queries

    def attack(self, batch_idx, x_0, x_b, true_label, target_label, sub_basis_torch, q_opt, iteration):
        q_num = 0
        grad = 0
        success_stop_queries = 0
        x_adv = x_b
        for i in range(iteration):
            grad_oi, ratios, num_call = self.black_grad_batch(x_b, sub_basis_torch, q_opt[i], self.sigma,
                                                    self.grad_estimator_batch_size, true_label, target_label)
            # q_num = q_num + q_opt[i]
            q_num = q_num + num_call
            grad = grad_oi + grad
            # dist = torch.norm((x_adv - x_0).view(self.batch_size, -1), self.ord, 1)
            x_adv, qs, eps = self.go_to_boundary(x_0, true_label, target_label, grad)
            q_num = q_num + qs
            x_adv, bin_query = self.bin_search(x_0,x_adv,true_label, target_label, self.tol)
            q_num = q_num + bin_query
            success_stop_queries = self.calculate_distortion(x_adv, x_0, success_stop_queries, q_num, batch_idx)
            log.info("{}-th image, distortion: {:.4f}, query:{}, iteration:{} ".format(batch_idx, torch.norm((x_adv - x_0).view(x_adv.size(0), -1), self.ord, 1).item(), q_num, i))
            x_b = x_adv

        x_adv = clip_image_values(x_adv, self.clip_min,self.clip_max)
        if success_stop_queries>self.maximum_queries:
            success_stop_queries = self.maximum_queries
        final_dist = torch.norm((x_adv - x_0).view(x_adv.size(0), -1), self.ord, 1)
        return x_adv, q_num, grad, success_stop_queries, final_dist, (final_dist <= self.epsilon)


    def opt_query_iteration(self, Nq, T, eta):
        coefs = [eta**(-2*i/3) for i in range(0,T)]
        sum_coefs = sum(coefs)
        opt_q=[round(Nq * coefs[i]/sum_coefs) for i in range(0,T)]

        if opt_q[0] > 80:
            T = T + 1
            opt_q, T = self.opt_query_iteration(Nq, T, eta)
        elif opt_q[0] < 50:
            T = T - 1
            opt_q, T = self.opt_query_iteration(Nq, T, eta)

        return opt_q, T

    def uni_query(self, Nq, T):
        opt_q = [round(Nq / T) for i in range(0, T)]
        return opt_q

    # def sparse_GeoDA(self, model, x_0, x_adv, true_label, gradient):
    #     list_coeff = np.linspace(0, 12, 36)
    #
    #     if self.norm == 'l1':
    #
    #         for delt in list_coeff:
    #             grad_l2 = gradient[0, :, :, :] / torch.norm(gradient[0, :, :, :])
    #             mint = 0
    #             maxt = 3 * 224 * 224
    #             dist = norm_inv_opt
    #
    #             multip = delt / np.sqrt(dist)
    #             for q_ind in range(20):
    #
    #                 mid = round((mint + maxt) / 2)
    #
    #                 grad_sp = self.topk_3D(grad_l2, mid)
    #                 grad_sp = grad_sp[None, :, :, :].cuda()
    #                 image_perturbed = x_0 + 5 * grad_sp
    #                 perturbed_clip = clip_image_values(image_perturbed, self.clip_min, self.clip_max)
    #
    #                 pert = clip_image_values(100 * grad_sp, self.clip_min, self.clip_max)
    #
    #                 x_B = x_0 + multip * (x_adv - x_0 / torch.norm(x_adv - x_0))
    #
    #                 grad_flatten = grad_l2.cpu().numpy().reshape(-1)
    #                 vec = perturbed_clip - x_B
    #                 vec_flatten = vec.cpu().numpy().reshape(-1)
    #
    #                 hyperplane = np.inner(grad_flatten, vec_flatten)
    #
    #                 if hyperplane > 0:
    #                     maxt = mid + 1
    #                 else:
    #                     mint = mid - 1
    #                 if maxt < 80:
    #
    #                     if self.is_adversarial(perturbed_clip, true_label) == False:
    #                         mid = maxt + int(maxt / 2) + 2
    #                         grad_sp = self.topk_3D(grad_l2, mid)
    #                         grad_sp = grad_sp[None, :, :, :].to(device)
    #                         image_perturbed = x_0 + 100 * grad_sp
    #                         perturbed_clip = clip_image_values(image_perturbed, lb, ub)
    #
    #                     break
    #
    #             if self.is_adversarial(perturbed_clip, true_label) == True:
    #                 print('Sparse perturbation is found.')
    #                 break
    #
    #         sparse_01 = inv_tf(perturbed_clip.cpu().numpy()[0, :, :, :].squeeze(), mean, std)
    #
    #         adv_label = torch.argmax(net.forward(Variable(perturbed_clip, requires_grad=True)).data).item()
    #         str_label_adv = get_label(labels[np.int(adv_label)].split(',')[0])
    #
    #         np.count_nonzero(abs(sparse_01 - image_fb))
    #         np.count_nonzero(abs((x_0 - perturbed_clip).cpu().numpy()))
    #         test = sparse_01 - image_fb
    #         test_torch = torch.tensor(test).to(device)
    #         grad_sp = self.topk_3D(test_torch, 1908)
    #
    #         ff = pert.cpu().numpy()
    #
    #
    #     if self.norm == 'l2' or self.norm == 'linf':
    #         adv_label = torch.argmax(model(x_adv),dim=1).item()
    #         str_label_adv = get_label(labels[np.int(adv_label)].split(',')[0])
    #
    #         pert_norm = abs(x_opt_inverse - image_fb) / np.linalg.norm(abs(x_opt_inverse - image_fb))
    #
    #         pert_norm_abs = (x_opt_inverse - image_fb) / np.linalg.norm((x_opt_inverse - image_fb))
    #
    #         pertimage = image_fb + 30 * pert_norm_abs


    def initialize_sub_basis(self, sub_dim):
        sub_basis = generate_2d_dct_basis(PROJECT_PATH, self.height,sub_dim).astype(np.float32)
        estimate_batch = self.grad_estimator_batch_size
        sub_basis_torch = torch.from_numpy(sub_basis).cuda()
        # EstNoise = SubNoise(estimate_batch, sub_basis_torch, self.channels, self.height, self.width)
        return sub_basis_torch

    def attack_all_images(self, args, arch_name, result_dump_path):
        sub_basis_torch = self.initialize_sub_basis(self.sub_dim)
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))
            not_done = correct.clone()
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
                target_label = target_labels[0].item()
            else:
                target_labels = None
                target_label = None

            true_label = true_labels[0].item()
            # log.info("before find_random_adversarial")
            if args.targeted:
                image_random = self.get_image_of_target_class(self.dataset, target_label, self.model)
                query_random_1 = 0
            else:
                image_random, query_random_1 = self.find_random_adversarial(images, true_label, None)
            # assert self.is_adversarial(image_random, true_label) == 1
            # Binary search
            # log.info("after find_random_adversarial")
            # log.info("before bin_search")
            x_boundary, query_binsearch_2 = self.bin_search(images, image_random, true_label, target_label, self.tol)
            x_b = x_boundary
            # log.info("after bin_search")
            # assert self.is_adversarial(x_boundary, true_label) == 1
            tot_query_rnd = query_binsearch_2 + query_random_1

            iteration = round(self.maximum_queries / 500)
            q_opt_it = int(self.maximum_queries - iteration * 25)
            q_opt_iter, iterate = self.opt_query_iteration(q_opt_it, iteration, self.mu)
            # log.info('#################################################################')
            # log.info('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(
            #     self.max_queries) + ', Norm = ' + self.norm + ', Space = ' + str(self.search_space))
            # log.info('#################################################################')

            adv_images, query_o, gradient, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, x_b, true_label,target_label,
                                                                                                                     sub_basis_torch, q_opt_iter, iterate)
            query = tot_query_rnd + query_o

            query = torch.tensor([query])
            success_query = torch.tensor([success_query])
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            with torch.no_grad():
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            # success = success_epsilon.float() * (success_query <= self.maximum_queries).float()
            # not_done = torch.ones_like(success) - success
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1-not_done) * correct * success_epsilon.float().cuda() * (success_query.cuda() <= self.maximum_queries).float()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries"]:
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
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "success_all":self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))



def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'GeoDA_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'GeoDA-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/GeoDA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument("--num_iterations",type=int,default=64)
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str, default="")
    parser.add_argument('--sub_dim', type=int)
    parser.add_argument('--max_queries',type=int, default=10000)

    args = parser.parse_args()
    assert args.batch_size == 1, "GeoDA only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted and args.dataset == "ImageNet":
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
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
        attacker = GeoDA(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                         args.norm,args.epsilon, search_space='sub',sub_dim=args.sub_dim,max_queries=args.max_queries,batch_size=args.batch_size)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()



