from collections import OrderedDict, defaultdict

import json
import torch
from torch.nn import functional as F
import numpy as np
import glog as log

from config import CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker

class SignOptL2Norm(object):
    def __init__(self, model, dataset, epsilon, targeted, batch_size=1, k=200, alpha=0.2, beta=0.001, iterations=1000,
                 maximum_queries=10000, svm=False, momentum=0.0, stopping=0.0001):
        self.model = model
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.maximum_queries = maximum_queries
        self.svm = svm
        self.momentum = momentum
        self.stopping = stopping
        self.epsilon  = epsilon
        self.targeted = targeted

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

    def fine_grained_binary_search_local(self,  x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 1
        lbd = initial_lbd

        # still inside boundary
        # log.info("size :{}".format((x0 + lbd * theta).size(0)))
        if self.model((x0 + lbd * theta).cuda()).max(1)[1].item() == y0:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while self.model((x0 + lbd_hi * theta).cuda()).max(1)[1].item() == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            # log.info("size :{}".format((x0 + lbd_lo * theta).size(0)))
            while self.model((x0 + lbd_lo * theta).cuda()).max(1)[1].item() != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            # log.info("size :{}".format((x0 + lbd_mid * theta).size(0)))
            if self.model((x0 + lbd_mid * theta).cuda()).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self,  x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            nquery += 1
            if self.model((x0 + current_best * theta).cuda()).max(1)[1].item() == y0:
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if self.model((x0 + lbd_mid * theta).cuda()).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def quad_solver(self, Q, b):
        """
        Solve min_a  0.5*aQa + b^T a s.t. a>=0
        """
        K = Q.size(0)
        alpha = torch.zeros(K)
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
        X = torch.zeros(dim, K)
        for iii in range(K):
            u = torch.randn_like(theta)
            u /= torch.norm(u)

            sign = 1
            new_theta = theta + h * u
            new_theta /= torch.norm(new_theta)

            # Targeted case.
            if (target_label is not None and
                    self.model((images + initial_lbd * new_theta).cuda()).max(1)[1].item() == target_label):
                sign = -1

            # Untargeted case

            if (target_label is None and
                    self.model((images + initial_lbd * new_theta).cuda()).max(1)[1].item() != true_label):
                sign = -1
            queries += 1
            X[:, iii] = sign * u.view(dim)

        Q = torch.matmul(X.transpose(0,1),X)
        q = -1 * torch.ones((K,))
        G = torch.diag(-1 * torch.ones((K,)))
        h = torch.zeros((K,))
        ### Use quad_qp solver
        # alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = self.quad_solver(Q, q)
        sign_grad = torch.matmul(X, alpha).view_as(theta)
        return sign_grad, queries

    def sign_grad_v1(self, images, true_label, theta, initial_lbd, h=0.001, target_label=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k  # 200 random directions (for estimating the gradient)
        sign_grad = torch.zeros_like(theta)
        queries = 0
        ### USe orthogonal transform
        # dim = np.prod(sign_grad.shape)
        # H = np.random.randn(dim, K)
        # Q, R = qr(H, mode='economic')
        for iii in range(K):  # for each u
            # # Code for reduced dimension gradient
            # u = np.random.randn(N_d,N_d)
            # u = u.repeat(D, axis=0).repeat(D, axis=1)
            # u /= LA.norm(u)
            # u = u.reshape([1,1,N,N])
            u = torch.randn_like(theta)
            u /= torch.norm(u)
            new_theta = theta + h * u
            new_theta /= torch.norm(new_theta)
            sign = 1

            # Targeted case.
            if (target_label is not None and
                    self.model((images + initial_lbd * new_theta).cuda()).max(1)[1].item() == target_label):
                sign = -1

            # Untargeted case
            if (target_label is None and
                    self.model((images + initial_lbd * new_theta).cuda()).max(1)[1].item() != true_label):  # success
                sign = -1

            queries += 1
            sign_grad += u * sign

        sign_grad /= K

        return sign_grad, queries


    def untargeted_attack(self, image_index, images, true_labels,):
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
        log.info("Searching for the initial direction on {} random directions: ".format(num_directions))
        for i in range(num_directions):
            query += 1
            theta = torch.randn_like(images)
            if self.model((images + theta).cuda()).max(1)[1].item() != true_label:
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
        prev_obj = 100000
        distortions = [gg]
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

                new_g2, count = self.fine_grained_binary_search_local(images, true_label, new_theta,
                                                                      initial_lbd=min_g2, tol=beta/500)
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
                    new_g2, count = self.fine_grained_binary_search_local(images, true_label, new_theta,
                                                                          initial_lbd=min_g2, tol=beta/500)
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
            distortions.append(gg)
            ## logging
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index+1, i+1, gg, query[0].item()))
            if query.min().item() >= self.maximum_queries:
                break
            
        if self.epsilon is None or gg <= self.epsilon:
            target = self.model((images + gg * xg).cuda()).max(1)[1].item()
            log.info("{}-th image success distortion {:.4f} target {} queries {} LS queries {}".format(image_index+1,
                                                                                                       gg, target, query[0].item(), ls_total))

            # FIXME 即使epsilon到了也不停止query，继续攻击
            # return images + gg * xg, gg, True, query, xg
        # gg 是distortion
        distortion = torch.norm(gg * xg, p=2)
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg.item(), distortion.item())

        return images + gg * xg, query,success_stop_queries, torch.tensor([gg]).float(), torch.tensor([gg]).float() <= self.epsilon, xg

    # def attack_targeted(self, x0, y0, target, alpha=0.2, beta=0.001, iterations=5000, query_limit=40000,
    #                     distortion=None, seed=None, svm=False, stopping=0.0001):
    #     """ Attack the original image and return adversarial example
    #         model: (pytorch model)
    #         train_dataset: set of training data
    #         (x0, y0): original image
    #     """
    #     model = self.model
    #     y0 = y0[0]
    #     print("Targeted attack - Source: {0} and Target: {1}".format(y0, target.item()))
    #
    #     if (model.predict_label(x0) == target):
    #         print("Image already target. No need to attack.")
    #         return x0, 0.0
    #
    #     if self.train_dataset is None:
    #         print("Need training dataset for initial theta.")
    #         return x0, 0.0
    #
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     num_samples = 100
    #     best_theta, g_theta = None, float('inf')
    #     query_count = 0
    #     ls_total = 0
    #     sample_count = 0
    #     print("Searching for the initial direction on %d samples: " % (num_samples))
    #     timestart = time.time()
    #
    #     #         samples = set(random.sample(range(len(self.train_dataset)), num_samples))
    #     #         print(samples)
    #     #         train_dataset = self.train_dataset[samples]
    #
    #     # Iterate through training dataset. Find best initial point for gradient descent.
    #     for i, (xi, yi) in enumerate(self.train_dataset):
    #         yi_pred = model.predict_label(xi.cuda())
    #         query_count += 1
    #         if yi_pred != target:
    #             continue
    #
    #         theta = xi.cpu().numpy() - x0.cpu().numpy()
    #         initial_lbd = LA.norm(theta)
    #         theta /= initial_lbd
    #         lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd,
    #                                                               g_theta)
    #         query_count += count
    #         if lbd < g_theta:
    #             best_theta, g_theta = theta, lbd
    #             print("--------> Found distortion %.4f" % g_theta)
    #
    #         sample_count += 1
    #         if sample_count >= num_samples:
    #             break
    #
    #         if i > 500:
    #             break
    #
    #     #         xi = initial_xi
    #     #         xi = xi.numpy()
    #     #         theta = xi - x0
    #     #         initial_lbd = LA.norm(theta.flatten(),np.inf)
    #     #         theta /= initial_lbd     # might have problem on the defination of direction
    #     #         lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, theta)
    #     #         query_count += count
    #     #         if lbd < g_theta:
    #     #             best_theta, g_theta = theta, lbd
    #     #             print("--------> Found distortion %.4f" % g_theta)
    #
    #     timeend = time.time()
    #     if g_theta == np.inf:
    #         return x0, float('inf')
    #     print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
    #           (g_theta, timeend - timestart, query_count))
    #
    #     # Begin Gradient Descent.
    #     timestart = time.time()
    #     xg, gg = best_theta, g_theta
    #     learning_rate = start_learning_rate
    #     prev_obj = 100000
    #     distortions = [gg]
    #     for i in range(iterations):
    #         if svm == True:
    #             sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta, target=target)
    #         else:
    #             sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta, target=target)
    #
    #         if False:
    #             # Compare cosine distance with numerical gradient.
    #             gradient, _ = self.eval_grad(model, x0, y0, xg, initial_lbd=gg, tol=beta / 500, h=0.01)
    #             print("    Numerical - Sign gradient cosine distance: ",
    #                   scipy.spatial.distance.cosine(gradient.flatten(), sign_gradient.flatten()))
    #
    #         # Line search
    #         ls_count = 0
    #         min_theta = xg
    #         min_g2 = gg
    #         for _ in range(15):
    #             new_theta = xg - alpha * sign_gradient
    #             new_theta /= LA.norm(new_theta)
    #             new_g2, count = self.fine_grained_binary_search_local_targeted(
    #                 model, x0, y0, target, new_theta, initial_lbd=min_g2, tol=beta / 500)
    #             ls_count += count
    #             alpha = alpha * 2
    #             if new_g2 < min_g2:
    #                 min_theta = new_theta
    #                 min_g2 = new_g2
    #             else:
    #                 break
    #
    #         if min_g2 >= gg:
    #             for _ in range(15):
    #                 alpha = alpha * 0.25
    #                 new_theta = xg - alpha * sign_gradient
    #                 new_theta /= LA.norm(new_theta)
    #                 new_g2, count = self.fine_grained_binary_search_local_targeted(
    #                     model, x0, y0, target, new_theta, initial_lbd=min_g2, tol=beta / 500)
    #                 ls_count += count
    #                 if new_g2 < gg:
    #                     min_theta = new_theta
    #                     min_g2 = new_g2
    #                     break
    #
    #         if alpha < 1e-4:
    #             alpha = 1.0
    #             print("Warning: not moving")
    #             beta = beta * 0.1
    #             if (beta < 1e-8):
    #                 break
    #
    #         xg, gg = min_theta, min_g2
    #
    #         query_count += (grad_queries + ls_count)
    #         ls_total += ls_count
    #         distortions.append(gg)
    #
    #         if query_count > query_limit:
    #             break
    #
    #         if i % 5 == 0:
    #             print("Iteration %3d distortion %.4f num_queries %d" % (i + 1, gg, query_count))
    #     #                 print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count,
    #     #                       " LR: ", alpha, "grad_queries", grad_queries, "ls_queries", ls_count)
    #
    #     # if distortion is not None and gg < distortion:
    #     #    print("Success: required distortion reached")
    #     #    break
    #
    #     #             if gg > prev_obj-stopping:
    #     #                 print("Success: stopping threshold reached")
    #     #                 break
    #     #             prev_obj = gg
    #
    #     adv_target = model.predict_label(x0 + torch.tensor(gg * xg, dtype=torch.float).cuda())
    #     if (adv_target == target):
    #         timeend = time.time()
    #         print("\nAdversarial Example Found Successfully: distortion %.4f target"
    #               " %d queries %d LS queries %d \nTime: %.4f seconds" % (
    #               gg, target, query_count, ls_total, timeend - timestart))
    #
    #         return x0 + torch.tensor(gg * xg, dtype=torch.float).cuda()
    #     else:
    #         print("Failed to find targeted adversarial example.")
    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):
        dist = torch.norm((perturbed - images).view(images.size(0), -1), p=2, dim=1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

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
            else:
                target_labels = None
            # return images + gg * xg, query,success_stop_queries, gg, gg <= self.epsilon, xg
            # adv, distortion, is_success, nqueries, theta_signopt
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
