import torch
import numpy as np
from scipy import fft
import random
import math
from typing import Callable, Union, Optional, Tuple, List, Any, Dict
from SurFree.utils import *

class SurFree(object):

    def __init__(
            self,
            model,
            steps: int = 100,
            norm :str = "l2",
            max_queries: int = 10000,
            BS_gamma: float = 0.01,
            BS_max_iteration: int = 10,
            theta_max: float = 30,
            n_ortho: int = 100,
            rho: float = 0.98,
            T: int = 3,
            with_alpha_line_search: bool = True,
            with_distance_line_search: bool = False,
            with_interpolation: bool = False):
        """
        Args:
            steps (int, optional): run steps. Defaults to 1000.
            max_queries (int, optional): stop running when each example require max_queries.
            BS_gamma ([type], optional): Binary Search Early Stop. Stop when precision is below BS_gamma. Defaults to 0.01.
            BS_max_iteration ([type], optional): Max iteration for . Defaults to 10.
            theta_max (int, optional): max theta watched to evaluate the direction. Defaults to 30.
            evolution (str, optional): Move in this direction. It can be linear or circular. Defaults to "circular".
            n_ortho (int, optional): Orthogonalize with the n last directions. Defaults to 100.
            rho (float, optional): Malus/Bonus factor given to the theta_max. Defaults to 0.98.
            T (int, optional): How many evaluation done to evaluated a direction. Defaults to 1.
            with_alpha_line_search (bool, optional): Activate Binary Search on Theta. Defaults to True.
            with_distance_line_search (bool, optional): Activate Binary Search between adversarial and x_o. Defaults to False.
            with_interpolation (bool, optional): Activate Interpolation. Defaults to False.
        """
        # Attack Parameters
        self.model = model
        self.norm = norm
        self._BS_gamma = BS_gamma
        self._BS_max_iteration = BS_max_iteration
        self._steps = steps
        self._max_queries = max_queries
        self.best_advs = None
        self._theta_max = theta_max
        self.rho = rho
        self.T = T
        assert self.rho <= 1 and self.rho > 0
        # Add or remove some parts of the attack
        self.with_alpha_line_search = with_alpha_line_search
        self.with_distance_line_search = with_distance_line_search
        self.with_interpolation = with_interpolation
        if self.with_interpolation and not self.with_distance_line_search:
            Warning("It's higly recommended to use Interpolation with distance line search.")

        # Data saved during attack
        self.n_ortho = n_ortho
        self._directions_ortho: Dict[int, torch.tensor] = {}
        self._nqueries: Dict[int, int] = {}

    def get_nqueries(self) -> Dict:
        return self._nqueries

    def compute_distance(self, x_ori, x_pert):
        # Compute the distance between two images.
        if self.norm == 'l2':
            return torch.norm(x_ori - x_pert,p=2)
        elif self.norm == 'linf':
            return torch.max(torch.abs(x_ori - x_pert))

    def run(
            self,
            images,
            criterion,
            *,
            early_stop: Optional[float] = None,
            starting_points: Optional[torch.tensor] = None,
            **kwargs: Any):

        self._nqueries = {i: 0 for i in range(len(images))}
        self._set_cos_sin_function(images)
        self.theta_max = torch.ones(images.size(0)) * self._theta_max
        criterion = get_criterion(criterion)
        self._criterion_is_adversarial = get_is_adversarial(criterion, model)

        # Get Starting Point
        if starting_points is not None:
            best_advs = starting_points
        elif starting_points is None:
            init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
            best_advs = init_attack.run(images, criterion, early_stop=early_stop)
        else:
            raise ValueError("starting_points {} doesn't exist.".format(starting_points))

        assert self._is_adversarial(best_advs).all()
        # Initialize the direction orthogonalized with the first direction
        fd = best_advs - images
        norm = torch.norm(fd.view(1,-1),p=2,dim=1)
        fd = fd / atleast_kd(norm, fd.dim())
        self._directions_ortho = {i: v.unsqueeze(0) for i, v in enumerate(fd)}

        # Load Basis
        if "basis_params" in kwargs:
            self._basis = Basis(images, **kwargs["basis_params"])
        else:
            self._basis = Basis(images)

        for _ in range(self._steps):
            # Get candidates. Shape: (n_candidates, batch_size, image_size)
            candidates = self._get_candidates(images, best_advs)
            candidates = candidates.permute(1, 0, 2, 3, 4)

            best_candidates = torch.zeros_like(best_advs)
            for i, o in enumerate(images):
                o_repeated = torch.cat([o.unsqueeze(0)] * len(candidates[i]), dim=0)
                index = torch.argmax(self.compute_distance(o_repeated, candidates[i]))
                best_candidates[i] = candidates[i][index]

            is_success = self.compute_distance(best_candidates, images) < self.compute_distance(best_advs, images)
            best_advs = torch.where(atleast_kd(is_success, best_candidates.dim()), torch.tensor(best_candidates), best_advs)

            if all(v > self._max_queries for v in self._nqueries.values()):
                print("Max queries attained for all the images.")
                break

        return best_advs

    def _is_adversarial(self, perturbed: torch.Tensor) -> torch.Tensor:
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1
        return self._criterion_is_adversarial(perturbed)

    def _get_candidates(self, originals: torch.Tensor, best_advs: torch.Tensor) -> torch.Tensor:
        """
        Find the lowest epsilon to misclassified x following the direction: q of class 1 / q + eps*direction of class 0
        """
        epsilons = torch.zeros(originals.size(0))
        direction_2 = torch.zeros_like(originals)
        while (epsilons == 0).any():
            direction_2 = torch.where(
                atleast_kd(epsilons == 0, direction_2.dim()),
                self._basis.get_vector(self._directions_ortho),
                direction_2
            )

            for i, eps_i in enumerate(epsilons):
                if eps_i == 0:
                    # Concatenate the first directions and the last directions generated
                    self._directions_ortho[i] = torch.cat((
                        self._directions_ortho[i][:1],
                        self._directions_ortho[i][1 + len(self._directions_ortho[i]) - self.n_ortho:],
                        direction_2[i].unsqueeze(0)), dim=0)

            function_evolution = self._get_evolution_function(originals, best_advs, direction_2)
            new_epsilons = self._get_best_theta(originals, function_evolution, epsilons)

            self.theta_max = torch.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = torch.where((new_epsilons != 0) * (epsilons == 0), self.theta_max / self.rho, self.theta_max)
            epsilons = new_epsilons

        epsilons = epsilons.unsqueeze(0)
        if self.with_interpolation:
            epsilons = torch.cat((epsilons, epsilons[0] / 2), dim=0)

        candidates = torch.cat([function_evolution(eps).unsqueeze(0) for eps in epsilons], dim=0)

        if self.with_interpolation:
            d = self.compute_distance(best_advs, originals)
            delta = self.compute_distance(self._binary_search(originals, candidates[1], boost=True), originals)
            theta_star = epsilons[0]

            num = theta_star * (4 * delta - d * (self._cos(theta_star) + 3))
            den = 4 * (2 * delta - d * (self._cos(theta_star) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            if self.with_distance_line_search:
                q_interp = self._binary_search(originals, q_interp, boost=True)
            candidates = torch.cat((candidates, q_interp.unsqueeze(0)), dim=0)

        return candidates

    def _get_evolution_function(self, originals: torch.Tensor, best_advs: torch.Tensor, direction_2: torch.Tensor) -> Callable[
        [torch.Tensor], torch.Tensor]:
        distances = self.compute_distance(best_advs, originals)
        direction_1 = (best_advs - originals).view(1,-1) / distances.view(-1, 1)
        direction_1 = direction_1.view_as(originals)
        return lambda theta: (originals + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta)).clamp(
            0, 1)

    def _get_best_theta(
            self,
            originals: torch.Tensor,
            function_evolution: Callable[[torch.Tensor], torch.Tensor],
            best_params: torch.Tensor) -> torch.Tensor:
        coefficients = torch.zeros(2 * self.T)
        for i in range(0, self.T):
            coefficients[2 * i] = 1 - (i / self.T)
            coefficients[2 * i + 1] = - coefficients[2 * i]

        for i, coeff in enumerate(coefficients):
            params = coeff * self.theta_max
            params = torch.where(torch.tensor(best_params == 0), params, torch.zeros_like(params))
            x = function_evolution(params)
            is_advs = self._is_adversarial(x)
            best_params = torch.where(
                torch.logical_and(best_params == 0, is_advs),
                params,
                best_params
            )
        if (best_params == 0).all() or not self.with_alpha_line_search:
            return best_params
        else:
            return self._alpha_binary_search(function_evolution, best_params, best_params != 0)

    def _alpha_binary_search(
            self,
            function_evolution: Callable[[torch.Tensor], torch.Tensor],
            lower: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
        # Upper --> not adversarial /  Lower --> adversarial

        def get_alpha(theta: torch.Tensor) -> torch.Tensor:
            return 1 - self._cos(theta * np.pi / 180)

        check_opposite = lower > 0  # if param < 0: abs(param) doesn't work

        # Get the upper range
        upper = torch.where(
            torch.logical_and(abs(lower) != self.theta_max, mask),
            lower + torch.sign(lower) * self.theta_max / self.T,
            torch.zeros_like(lower)
        )

        mask_upper = torch.logical_and(upper == 0, mask)
        while mask_upper.any():
            # Find the correct lower/upper range
            upper = torch.where(
                mask_upper,
                lower + torch.sign(lower) * self.theta_max / self.T,
                upper
            )
            x = function_evolution(upper)

            mask_upper = mask_upper * self._is_adversarial(x)
            lower = torch.where(mask_upper, upper, lower)

        step = 0
        while step < self._BS_max_iteration and (abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma).any():
            mid_bound = (upper + lower) / 2
            mid = function_evolution(mid_bound)
            is_adv = self._is_adversarial(mid)

            mid_opp = torch.where(
                atleast_kd(torch.tensor(check_opposite), mid.dim()),
                function_evolution(-mid_bound),
                torch.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = torch.where(mask * is_adv, mid_bound, lower)
            lower = torch.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound, lower)
            upper = torch.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)
            upper = torch.where(mask * abs(lower) != abs(mid_bound), mid_bound, upper)

            check_opposite = mask * check_opposite * is_adv_opp * (lower > 0)

            step += 1
        return torch.tensor(lower)

    def _binary_search(self, originals: torch.Tensor, perturbed: torch.Tensor, boost: Optional[bool] = False) -> ep.Tensor:
        # Choose upper thresholds in binary search based on constraint.
        highs = torch.ones(perturbed.size(0))
        d = np.prod(list(perturbed.size())[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = torch.zeros_like(highs)

        # Boost Binary search
        if boost:
            boost_vec = 0.1 * originals + 0.9 * perturbed
            is_advs = self._is_adversarial(boost_vec)
            is_advs = atleast_kd(is_advs, originals.dim())
            originals = torch.where(is_advs.logical_not(), boost_vec, originals)
            perturbed = torch.where(is_advs, boost_vec, perturbed)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs
        iteration = 0
        while torch.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = self._is_adversarial(mids_perturbed)

            highs = torch.where(is_adversarial_, mids, highs)
            lows = torch.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break

        results = self._project(originals, perturbed, highs)
        return results

    def _project(self, originals: torch.Tensor, perturbed: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        epsilons = atleast_kd(epsilons, originals.dim())
        return (1.0 - epsilons) * originals + epsilons * perturbed

    def _add_step_in_circular_direction(self, direction1: torch.Tensor, direction2: torch.Tensor, r: torch.Tensor,
                                        degree: torch.Tensor) -> torch.Tensor:
        degree = atleast_kd(degree, direction1.dim())
        r = atleast_kd(r, direction1.dim())
        results = self._cos(degree * np.pi / 180) * direction1 + self._sin(degree * np.pi / 180) * direction2
        results = results * r * self._cos(degree * np.pi / 180)
        return torch.tensor(results)

    def _set_cos_sin_function(self, v: torch.Tensor) -> None:
        self._cos, self._sin = torch.cos, torch.sin


class Basis:
    def __init__(self, originals: torch.Tensor, random_noise: str = "normal", basis_type: str = "dct", **kwargs: Any):
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or
                                          uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * alpha
                            * beta
                            * lambda
                            * frequence_range: integers or float
                            * min_dct_value
                            * dct_type: 8x8 or full
        """
        self._originals = originals
        self._direction_shape = originals.shape[1:]
        self.basis_type = basis_type

        self._load_params(**kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with: Optional[Dict] = None, bounds: Tuple[float, float] = (0, 1)) -> ep.Tensor:
        if ortho_with is None:
            ortho_with = {i: None for i in range(len(self._originals))}

        vectors = [
            self.get_vector_i(i, ortho_with[i], bounds).unsqueeze(0)
            for i in range(len(self._originals))
        ]
        return torch.cat(vectors, dim=0)

    def get_vector_i(self, index: int, ortho_with: Optional[torch.Tensor] = None,
                     bounds: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        r: torch.Tensor = getattr(self, "_get_vector_i_" + self.basis_type)(index, bounds)

        if ortho_with is not None:
            r_repeated = torch.cat([r.unsqueeze(0)] * len(ortho_with), dim=0)

            # inner product
            gs_coeff = (ortho_with * r_repeated).flatten(1).sum(1)
            proj = atleast_kd(gs_coeff, ortho_with.dim()) * ortho_with
            r = r - proj.sum(0)
        return r / torch.norm(r,p=2)

    def _get_vector_i_dct(self, index: int) -> torch.Tensor:
        r_np = np.zeros(self._direction_shape)
        for channel, dct_channel in enumerate(self.dcts[index]):
            probs = np.random.randint(-2, 1, dct_channel.shape) + 1
            r_np[channel] = dct_channel * probs
        r_np = idct2_8_8(r_np) + self._beta * (2 * np.random.rand(*r_np.shape) - 1)
        return torch.from_numpy(self._originals).type(torch.float32)

    def _get_vector_i_random(self, bounds: Tuple[float, float]) -> torch.Tensor:
        r = torch.zeros(self._direction_shape)
        r = getattr(torch, self.random_noise)(r, r.shape, *bounds)
        return torch.tensor(r)

    def _load_params(
            self,
            beta: float = 0,
            frequence_range: Tuple[float, float] = (0, 1),
            dct_type: str = "8x8",
            function: str = "tanh",
            lambda_: float = 1
    ) -> None:
        if not hasattr(self, "_get_vector_i_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = beta
            if dct_type == "8x8":
                mask_size = (8, 8)
                dct_function = dct2_8_8
            elif dct_type == "full":
                mask_size = (self._direction_shape[-2], self._direction_shape[-1])
                dct_function = dct2_8_8
            else:
                raise ValueError("DCT {} doesn't exist.".format(dct_type))

            dct_mask = get_zig_zag_mask(frequence_range, mask_size)
            self.dcts = np.array([dct_function(np.array(image.detach().cpu()), dct_mask) for image in self._originals])

            def get_function(function: str, lambda_: float) -> Callable:
                if function == "tanh":
                    return lambda x: np.tanh(lambda_ * x)
                elif function == "identity":
                    return lambda x: x
                elif function == "constant":
                    return lambda x: (abs(x) > 0).astype(int)
                else:
                    raise ValueError("Function given for DCT is incorrect.")

            self.dcts = get_function(function, lambda_)(self.dcts)


