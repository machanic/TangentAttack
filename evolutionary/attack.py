import os
import sys
from collections import defaultdict, OrderedDict, deque

import cv2
import torch
import tempfile
import numpy as np
# use MPI Spawn to start workers
from mpi4py import MPI

from dataset.dataset_loader_maker import DataLoaderMaker
import glog as log

class Evolutionary(object):
    ''' Evolutionary. A black-box decision-based method.
    - Supported distance metric: ``l_2``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1904.04433.
    '''

    def __init__(self, model, dataset, batch_size, clip_min, clip_max, targeted, maximum_queries, dimension_reduction=None):
        ''' Initialize Evolutionary.
        :param model: The model to attack. A ``realsafe.model.Classifier`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param dimension_reduction: ``(height, width)``.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''
        self.model, self.batch_size = model, batch_size
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.maximum_queries = maximum_queries
        self.dimension_reduction = dimension_reduction
        if self.dimension_reduction is not None:
            # to avoid import tensorflow in other processes, we cast the dimension to basic type
            self.dimension_reduction = (int(self.dimension_reduction[0]), int(self.dimension_reduction[1]))

        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)




    def split_trunks(self, xs, n):
        N = len(xs)
        trunks = []
        trunk_size = N // n
        if N % n == 0:
            for rank in range(n):
                start = rank * trunk_size
                trunks.append(xs[start:start + trunk_size])
        else:
            for rank in range(N % n):
                start = rank * (trunk_size + 1)
                trunks.append(xs[start:start + trunk_size + 1])
            for rank in range(N % n, n):
                start = rank * trunk_size + (N % n)
                trunks.append(xs[start:start + trunk_size])
        return trunks

    def config(self, **kwargs):
        ''' (Re)config the attack.
        :param starting_points: Starting points which are already adversarial. A numpy array with data type of
            ``self.x_dtype``, with shape of ``(self.batch_size, *self.x_shape)``.
        :param max_queries: Max queries. An integer.
        :param mu: A hyper-parameter controlling the mean of the Gaussian distribution. A float number.
        :param sigma: A hyper-parameter controlling the variance of the Gaussian distribution. A float number.
        :param decay_factor: The decay factor for the evolution path. A float number.
        :param c: The decay factor for the covariance matrix. A float number.
        :param maxprocs: Max number of processes to run MPI tasks. An Integer.
        :param logger: A standard logger for logging verbose information during attacking.
        '''
        if 'starting_points' in kwargs:
            self.starting_points = kwargs['starting_points']

        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']

        if 'mu' in kwargs:
            self.mu = kwargs['mu']
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'decay_factor' in kwargs:
            self.decay_factor = kwargs['decay_factor']
        if 'c' in kwargs:
            self.c = kwargs['c']

        if 'maxprocs' in kwargs:
            self.maxprocs = kwargs['maxprocs']

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def _batch_attack_generator(self, xs, ys, ys_target):
        ''' Attack a batch of examples. It is a generator which yields back ``iteration_callback()``'s return value
        after each iteration (query) if the ``iteration_callback`` is not ``None``, and returns the adversarial
        examples.
        '''
        if self.iteration_callback is not None:
            self._session.run(self.setup_xs_var, feed_dict={self.xs_ph: xs})
        # use named memmap to speed up IPC
        xs_shm_file = tempfile.NamedTemporaryFile(prefix='/dev/shm/realsafe_evolutionary_')
        xs_adv_shm_file = tempfile.NamedTemporaryFile(prefix='/dev/shm/realsafe_evolutionary_xs_adv_')
        xs_shm = np.memmap(xs_shm_file.name, dtype=self.model.x_dtype.as_numpy_dtype, mode='w+',
                           shape=(self.batch_size, *self.model.x_shape))
        xs_adv_shm = np.memmap(xs_adv_shm_file.name, dtype=self.model.x_dtype.as_numpy_dtype, mode='w+',
                               shape=(self.batch_size, *self.model.x_shape))

        # use a proper number of processes
        nprocs = self.batch_size if self.batch_size <= self.maxprocs else self.maxprocs
        # since we use memmap here, run everything on localhost
        info = MPI.Info.Create()
        info.Set("host", "localhost")
        # spawn workers
        worker = os.path.abspath(os.path.join(os.path.dirname(__file__), './evolutionary_worker.py'))
        comm = MPI.COMM_SELF.Spawn(sys.executable, maxprocs=nprocs, info=info,
                                   args=[worker, xs_shm_file.name, xs_adv_shm_file.name, str(self.batch_size)])
        # prepare shared arguments
        shared_args = {
            'x_dtype': self.model.x_dtype.as_numpy_dtype,  # avoid importing tensorflow in workers
            'x_shape': self.model.x_shape,
            'x_min': float(self.model.x_min),
            'x_max': float(self.model.x_max),
            'mu': float(self.mu),
            'sigma': float(self.sigma),
            'decay_factor': float(self.decay_factor),
            'c': float(self.c),
            'goal': self.goal,
            'dimension_reduction': self.dimension_reduction,
        }
        # prepare tasks
        all_tasks = []
        for i in range(self.batch_size):
            all_tasks.append({
                'index': i,
                'x': xs[i],
                'starting_point': self.starting_points[i],
                'y': None if ys is None else ys[i],
                'y_target': None if ys_target is None else ys_target[i],
            })
        # split tasks into trunks for each worker
        trunks = self.split_trunks(all_tasks, nprocs)
        # send arguments to workers
        comm.bcast(shared_args, root=MPI.ROOT)
        comm.scatter(trunks, root=MPI.ROOT)

        # the main loop
        for q in range(self.max_queries + 1):  # the first query is used to check the original examples
            # collect log from workers
            reqs = comm.gather(None, root=MPI.ROOT)
            if self.logger:
                for logs in reqs:
                    for log in logs:
                        self.logger.info(log)
            # yield back iteration_callback return value
            if self.iteration_callback is not None and q >= 1:
                yield self._session.run(self.iteration_callback, feed_dict={self.xs_ph: xs_adv_shm})
            if q == self.max_queries:
                # send a None to all workers, so that they could exit
                comm.scatter([None for _ in range(nprocs)], root=MPI.ROOT)
                reqs = comm.gather(None, root=MPI.ROOT)
                if self.logger:
                    for logs in reqs:
                        for log in logs:
                            self.logger.info(log)
            else:  # run predictions for xs_shm
                xs_ph_labels = self._session.run(self.xs_ph_labels, feed_dict={self.xs_ph: xs_shm})
                xs_ph_labels = xs_ph_labels.tolist()  # avoid pickle overhead of numpy array
                comm.scatter(self.split_trunks(xs_ph_labels, nprocs), root=MPI.ROOT)  # send predictions to workers
        # disconnect from MPI Spawn
        comm.Disconnect()
        # copy the xs_adv
        xs_adv = xs_adv_shm.copy()

        # delete the temp file
        xs_shm_file.close()
        xs_adv_shm_file.close()
        del xs_shm
        del xs_adv_shm

        return xs_adv

    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.
        :return: When the ``iteration_callback`` is ``None``, return the generated adversarial examples. When the
            ``iteration_callback`` is not ``None``, return a generator, which yields back the callback's return value
            after each iteration and returns the generated adversarial examples.
        '''
        g = self._batch_attack_generator(xs, ys, ys_target)
        if self.iteration_callback is None:
            try:
                next(g)
            except StopIteration as exp:
                return exp.value
        else:
            return g

    def attack(self, index, x, starting_point, y, y_target,
               x_dtype, x_shape, x_min, x_max,
               mu, sigma, decay_factor, c, dimension_reduction,
               logs, xs_adv_shm):

        def fn_is_adversarial(label):
            if not self.targeted:
                return label != y
            else:
                return label == y_target

        def fn_mean_square_distance(x1, x2):
            return np.mean((x1 - x2) ** 2) / ((x_max - x_min) ** 2)

        x_label = yield x
        if fn_is_adversarial(x_label):
            log.info('{}: The original image is already adversarial'.format(index))
            xs_adv_shm[index] = x
            return

        xs_adv_shm[index] = starting_point
        x_adv = starting_point
        dist = fn_mean_square_distance(x, x_adv)
        stats_adversarial = deque(maxlen=30)

        if dimension_reduction:
            assert len(x_shape) == 3
            pert_shape = (*dimension_reduction, x_shape[2])
        else:
            pert_shape = x_shape

        N = np.prod(pert_shape)
        K = int(N / 20)

        evolution_path = np.zeros(pert_shape, dtype=x_dtype)
        diagonal_covariance = np.ones(pert_shape, dtype=x_dtype)

        x_adv_label = yield x_adv

        log.info('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
            index, 0, dist, x_adv_label, sigma, mu, ''
        ))

        step = 0
        while True:
            step += 1
            unnormalized_source_direction = x - x_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)

            selection_probability = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

            perturbation = np.random.normal(0.0, 1.0, pert_shape).astype(x_dtype)
            factor = np.zeros([N], dtype=x_dtype)
            factor[selected_indices] = 1
            perturbation *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)

            if dimension_reduction:
                perturbation_large = cv2.resize(perturbation, x_shape[:2])
            else:
                perturbation_large = perturbation

            biased = x_adv + mu * unnormalized_source_direction
            candidate = biased + sigma * source_norm * perturbation_large / np.linalg.norm(perturbation_large)
            candidate = x - (x - candidate) / np.linalg.norm(x - candidate) * np.linalg.norm(x - biased)
            candidate = np.clip(candidate, x_min, x_max)

            candidate_label = yield candidate

            is_adversarial = fn_is_adversarial(candidate_label)
            stats_adversarial.appendleft(is_adversarial)

            if is_adversarial:
                xs_adv_shm[index] = candidate
                new_x_adv = candidate
                new_dist = fn_mean_square_distance(new_x_adv, x)
                evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation
                diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)
            else:
                new_x_adv = None

            message = ''
            if new_x_adv is not None:
                abs_improvement = dist - new_dist
                rel_improvement = abs_improvement / dist
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100, abs_improvement)
                x_adv, dist = new_x_adv, new_dist
                x_adv_label = candidate_label

            log.info('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
                index, step, dist, x_adv_label, sigma, mu, message))

            if len(stats_adversarial) == stats_adversarial.maxlen:
                p_step = np.mean(stats_adversarial)
                mu *= np.exp(p_step - 0.2)
                stats_adversarial.clear()