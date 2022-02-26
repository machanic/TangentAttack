from collections import OrderedDict, defaultdict

import json
import torch
from torch.nn import functional as F
import numpy as np
import glog as log

from config import CLASS_NUM, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, TinyImageNetDataset


class SignOptL2Norm(object):
    def __init__(self, model, dataset, epsilon, targeted, batch_size=1, k=200, alpha=0.2, beta=0.001, iterations=1000,
                 maximum_queries=10000, svm=False, momentum=0.0, tot=None, best_initial_target_sample=False):
        self.model = model
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.maximum_queries = maximum_queries
        self.svm = svm
        self.momentum = momentum
        self.epsilon  = epsilon
        self.targeted = targeted
        self.best_initial_target_sample = best_initial_target_sample
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
        self.tot = tot

    def fine_grained_binary_search_local(self,  x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 1
        lbd = initial_lbd

        # still inside boundary
        if self.model(x0 + lbd * theta).max(1)[1].item() == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.model(x0 + lbd_hi * theta).max(1)[1].item() == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.model(x0 + lbd_lo * theta).max(1)[1].item() != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1
        tot_count = 0
        while (lbd_hi - lbd_lo) > tol:
            tot_count+=1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if tot_count>200:
                log.info("reach max while limit, maybe dead loop in binary search function, break!")
                break
        return lbd_hi, nquery

    def fine_grained_binary_search_local_targeted(self, x0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 1
        lbd = initial_lbd

        if self.model(x0 + lbd * theta).max(1)[1].item() != t:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.model(x0 + lbd_hi * theta).max(1)[1].item() != t:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while self.model(x0 + lbd_lo * theta).max(1)[1].item() == t:
                lbd_lo = lbd_lo * 0.99
                nquery += 1
        tot_count = 0
        while (lbd_hi - lbd_lo) > tol:
            tot_count += 1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if tot_count>200:
                log.info("reach max while limit, dead loop in binary search function, break!")
                break
        return lbd_hi, nquery

    def fine_grained_binary_search(self,  x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            nquery += 1
            if self.model(x0 + current_best * theta).max(1)[1].item() == y0:
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            count+= 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if count >= 200:
                log.info("Break in the first fine_grained_binary_search!")
                break
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            nquery += 1
            if self.model(x0 + current_best * theta).max(1)[1].item() != t:
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            count += 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
            if count >= 200:
                log.info("Break in the first fine_grained_binary_search!")
                break
        return lbd_hi, nquery



    def quad_solver(self, Q, b):
        """
        Solve min_a  0.5*aQa + b^T a s.t. a>=0
        """
        K = Q.size(0)
        alpha = torch.zeros(K,device='cuda')
        g = b
        Qdiag = torch.diag(Q)
        for i in range(20000):
            delta = torch.clamp(alpha - g / Qdiag,min=0) - alpha
            idx = torch.argmax(torch.abs(delta))
            val = delta[idx].item()
            if abs(val) < 1e-7:
                break
            g = g + val * Q[:, idx]
            alpha[idx] += val
        return alpha

    def sign_grad_svm(self, images, true_label, theta, initial_lbd, h=0.001, K=100, target_label=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        queries = 0
        dim = np.prod(list(theta.size())).item()
        # X = torch.zeros(dim, K).cuda()
        images_batch = []
        u_batch = []

        for iii in range(K):
            u = torch.randn_like(theta)
            u /= torch.linalg.norm(u)

            # sign = 1
            new_theta = theta + h * u
            new_theta /= torch.linalg.norm(new_theta)
            images_batch.append(images + initial_lbd * new_theta)
            u_batch.append(u)
            queries += 1
            # delete below code to accelrate speed by batch
            # # Targeted case.
            # if (target_label is not None and
            #         self.model(images + initial_lbd * new_theta).max(1)[1].item() == target_label):
            #     sign = -1
            #
            # # Untargeted case
            #
            # if (target_label is None and
            #         self.model(images + initial_lbd * new_theta).max(1)[1].item() != true_label):
            #     sign = -1

            # X[:, iii] = sign * u.view(dim)
        # the batch image feed process
        images_batch = torch.cat(images_batch,0)
        u_batch = torch.cat(u_batch,0) # K,C,H,W
        sign = torch.ones(K,device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(K)],device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(K)],device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels!=true_labels] = -1

        X = torch.transpose(u_batch.view(K, dim) * sign.view(K,1),0,1)  # convert from X[:, iii] = sign * u.view(dim)
        Q = torch.mm(X.transpose(0,1),X)  # K,dim x dim,K = K,K
        q = -1 * torch.ones((K,),device='cuda')
        G = torch.diag(-1 * torch.ones((K,)))
        h = torch.zeros((K,))
        ### Use quad_qp solver
        # alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = self.quad_solver(Q, q) # K,K x K = K,1
        sign_grad = torch.matmul(X, alpha).view_as(theta)
        return sign_grad, queries

    # the version of accelerate speed by batch feeding
    def sign_grad_v1(self, images, true_label, theta, initial_lbd, h=0.001, target_label=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k  # 200 random directions (for estimating the gradient)
        # sign_grad = torch.zeros_like(theta)
        queries = 0
        ### USe orthogonal transform
        # dim = np.prod(sign_grad.shape)
        # H = np.random.randn(dim, K)
        # Q, R = qr(H, mode='economic')
        images_batch = []
        u_batch = []
        for iii in range(K):  # for each u
            # # Code for reduced dimension gradient
            # u = np.random.randn(N_d,N_d)
            # u = u.repeat(D, axis=0).repeat(D, axis=1)
            # u /= LA.norm(u)
            # u = u.reshape([1,1,N,N])
            u = torch.randn_like(theta)
            u /= torch.linalg.norm(u)
            new_theta = theta + h * u
            new_theta /= torch.linalg.norm(new_theta)
            # sign = 1
            u_batch.append(u)
            images_batch.append(images + initial_lbd * new_theta)
            # the below code is so slow, accelerate it by batch, so I delete it
            # # Targeted case.
            # if (target_label is not None and
            #         self.model(images + initial_lbd * new_theta).max(1)[1].item() == target_label):
            #     sign = -1
            #
            # # Untargeted case
            # if (target_label is None and
            #         self.model(images + initial_lbd * new_theta).max(1)[1].item() != true_label):  # success
            #     sign = -1
            # sign_grad += u * sign
            queries += 1
        images_batch = torch.cat(images_batch,0)
        u_batch = torch.cat(u_batch,0)  # B,C,H,W
        assert u_batch.dim() == 4
        sign = torch.ones(K,device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(K)],device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(K)],device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels!=true_labels] = -1
        sign_grad = torch.sum(u_batch * sign.view(K,1,1,1),dim=0,keepdim=True)

        sign_grad = sign_grad / K

        return sign_grad, queries

    def untargeted_attack(self, image_index, images, true_labels):
        assert images.size(0) == 1
        alpha = self.alpha
        beta = self.beta
        momentum = self.momentum
        batch_image_positions = np.arange(image_index * self.batch_size,
                                          min((image_index + 1) * self.batch_size, self.total_images)).tolist()
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        ls_total = 0
        true_label = true_labels[0].item()
        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        log.info("Searching for the initial direction on {} random directions.".format(num_directions))
        for i in range(num_directions):
            query += 1
            theta = torch.randn_like(images)
            if self.model(images + theta).max(1)[1].item() != true_label:
                initial_lbd = torch.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(images, true_label, theta, initial_lbd, g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    log.info("{}-th image, {}-th iteration distortion: {:.4f}".format(image_index+1, i, g_theta))
                    self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query, success_stop_queries,
                                                         batch_image_positions)
        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index+1))
            return images, query,success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index+1, g_theta,  query[0].item()))
        #### Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        vg = torch.zeros_like(xg)
        for i in range(self.iterations):
            ## gradient estimation at x0 + theta (init)
            if self.svm:
                sign_gradient, grad_queries = self.sign_grad_svm(images, true_label, xg, initial_lbd=gg, h=beta)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(images, true_label, xg, initial_lbd=gg, h=beta)
            ## Line search of the step size of gradient descent
            query += grad_queries
            ls_count = 0  # line search queries

            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta
            min_vg = vg  ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                if momentum > 0:
                    new_vg = momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta)
                tol = beta/500
                if self.tot is not None:
                    tol = self.tot
                new_g2, count = self.fine_grained_binary_search_local(images, true_label, new_theta,
                                                                      initial_lbd=min_g2, tol=tol)
                ls_count += count
                query += count

                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                         success_stop_queries, batch_image_positions)
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break
            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        new_vg = momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta)
                    tol = beta / 500
                    if self.tot is not None:
                        tol = self.tot
                    new_g2, count = self.fine_grained_binary_search_local(images, true_label, new_theta,
                                                                          initial_lbd=min_g2, tol=tol)
                    ls_count += count
                    query += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                             success_stop_queries, batch_image_positions)
                        if momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                log.info("{}-th image warns: not moving".format(image_index+1))
                beta = beta * 0.1
                if beta < 1e-8:
                    break
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            vg = min_vg

            ls_total += ls_count
            ## logging
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index+1, i+1, gg, query[0].item()))
            if query.min().item() >= self.maximum_queries:
                break
        if self.epsilon is None or gg <= self.epsilon:
            target = self.model(images + gg * xg).max(1)[1].item()
            log.info("{}-th image success distortion {:.4f} target {} queries {} LS queries {}".format(image_index+1,
                                                                                                       gg, target, query[0].item(), ls_total))
        # gg 是distortion
        distortion = torch.norm(gg * xg, p=2)
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        # success_stop_queries = torch.clamp(success_stop_queries,0,self.maximum_queries)
        return images + gg * xg, query,success_stop_queries, torch.tensor([gg]).float(), torch.tensor([gg]).float() <= self.epsilon, xg

    def get_image_of_target_class(self,dataset_name, target_labels, target_model):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "TinyImageNet":
                dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                      size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                      align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
            max_recursive_loop_limit = 100
            loop_count = 0
            while logits.max(1)[1].item() != label.item() and loop_count < max_recursive_loop_limit:
                loop_count += 1
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                          size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                          align_corners=False)
                with torch.no_grad():
                    logits = target_model(image.cuda())

            if loop_count == max_recursive_loop_limit:
                # The program cannot find a valid image from the validation set.
                return None

            assert true_label == label.item()
            images.append(torch.squeeze(image))
        return torch.stack(images).cuda()  # B,C,H,W


    def targeted_attack(self, image_index, images, target_labels, target_class_image):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        target_label = target_labels[0].item()

        if (self.model(images).max(1)[1].item() == target_label):
            log.info("{}=th image is already predicted as target label! No need to attack.".format(image_index+1))

        alpha = self.alpha
        beta = self.beta
        batch_image_positions = np.arange(image_index * self.batch_size,
                                          min((image_index + 1) * self.batch_size, self.total_images)).tolist()
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        ls_total = 0

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        log.info("Searching for the initial direction on {} samples: ".format(num_samples))
        if self.best_initial_target_sample:
            # Iterate through training dataset. Find best initial point for gradient descent.
            if self.dataset == "ImageNet":
                val_dataset = ImageNetDataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            elif self.dataset == "CIFAR-10":
                val_dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            elif self.dataset == "CIFAR-100":
                val_dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)
            for i, (xi, yi) in enumerate(val_dataset_loader):
                if self.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                    xi = F.interpolate(xi,
                                           size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                           align_corners=False)
                xi = xi.cuda()
                yi_pred = self.model(xi).max(1)[1].item()
                query += 1
                if yi_pred != target_label:
                    continue

                theta = xi - images
                initial_lbd = torch.linalg.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd,
                                                                      g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query,
                                                         success_stop_queries, batch_image_positions)
                    log.info("{}-th image. Found initial target image with the distortion {:.4f}".format(image_index+1, g_theta))

                if i > 100:
                    break
        else:
            # xi = self.get_image_of_target_class(self.dataset, target_labels, self.model)
            xi = target_class_image
            theta = xi - images
            initial_lbd = torch.linalg.norm(theta)
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd,
                                                                  g_theta)
            query += count
            best_theta, g_theta = theta, lbd
            self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query,
                                                 success_stop_queries, batch_image_positions)
        if g_theta == np.inf:
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index + 1))
            return images, query, success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index + 1, g_theta,
                                                                                    query[0].item()))
        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(self.iterations):
            if self.svm:
                sign_gradient, grad_queries = self.sign_grad_svm(images, None, xg, initial_lbd=gg, h=beta, target_label=target_label)
            else:
                sign_gradient, grad_queries = self.sign_grad_v1(images, None, xg, initial_lbd=gg, h=beta, target_label=target_label)
            query += grad_queries
            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= torch.linalg.norm(new_theta)
                tol = beta / 500
                if self.tot is not None:
                    tol = self.tot
                new_g2, count = self.fine_grained_binary_search_local_targeted(images, target_label, new_theta, initial_lbd=min_g2, tol=tol)
                ls_count += count
                query += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                         success_stop_queries, batch_image_positions)
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.linalg.norm(new_theta)
                    tol = beta / 500
                    if self.tot is not None:
                        tol = self.tot
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                         images, target_label, new_theta, initial_lbd=min_g2, tol=tol)
                    ls_count += count
                    query += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                             success_stop_queries, batch_image_positions)
                        break

            if alpha < 1e-4:
                alpha = 1.0
                log.info("{}-th image, warning: not moving".format(image_index+1))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_theta, min_g2

            ls_total += ls_count
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index + 1, i + 1, gg,
                                                                                           query[0].item()))
            if query.min().item() >= self.maximum_queries:
                break

        log.info(
            "{}-th image success distortion {:.4f} queries {} stop queries {}".format(image_index + 1,
                                                                                              gg,
                                                                                              query[0].item(),
                                                                                              success_stop_queries[0].item()))

        adv_target = self.model(images + gg * xg).max(1)[1].item()
        if adv_target == target_label:
            log.info("{}-th image attack successfully! Distortion {:.4f} target {} queries:{} success stop queries:{} LS queries:{}".format(image_index + 1,
                                                                                                       gg, adv_target,
                                                                                                       query[0].item(), success_stop_queries[0].item(),
                                                                                                       ls_total))
        else:
            log.info("{}-th image is failed to find targeted adversarial example.".format(image_index+1))

        distortion = torch.norm(gg * xg, p=2)
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        # success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return images + gg * xg, query, success_stop_queries, torch.tensor([gg]).float(), torch.tensor(
            [gg]).float() <= self.epsilon, xg



    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):
        dist = torch.norm((perturbed - images).view(images.size(0), -1), p=2, dim=1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

    def attack_all_images(self, args, arch_name, result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
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
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item()!=true_labels[0].item()
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            # return images + gg * xg, query,success_stop_queries, gg, gg <= self.epsilon, xg
            # adv, distortion, is_success, nqueries, theta_signopt
            if args.targeted:
                target_class_image = self.get_image_of_target_class(self.dataset, target_labels, self.model)
                if target_class_image is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta_signopt = self.targeted_attack(batch_index, images, target_labels, target_class_image)
            else:
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta_signopt = self.untargeted_attack(batch_index,
                                                                                                                    images,  true_labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()

            with torch.no_grad():
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * success_epsilon.float() *(success_query <= self.maximum_queries).float()

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
