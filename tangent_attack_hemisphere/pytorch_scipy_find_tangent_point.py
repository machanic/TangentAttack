import torch
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.sparse import identity, csr_matrix
import numpy as np
from torch.optim import SGD
import glog as log

class TangentPointFinder(object):
    def __init__(self, dim, initial_with_penalty_method, use_scipy,scipy_method="trust-constr",
                 clip_min=0, clip_max=1.0):
        self.use_scipy = use_scipy
        self.initial_with_penalty_method = initial_with_penalty_method
        self.dim = dim
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.bounds = [(self.clip_min, self.clip_max) for _ in range(dim)]
        self.scipy_method = scipy_method
        self.hess_of_constraint_cache_value = identity(dim,dtype=np.float32) * 2.0
        self.hess_of_objective_cache_value = csr_matrix((dim,dim),dtype=np.float32)
        self.hessp = np.zeros(dim,dtype=np.float32)

    def get_maximize_objective_function(self, x0, normal_vector):
        func = lambda x: np.dot(x-x0, normal_vector)  # minimize (negative sign)
        return func

    def get_maximize_jacobian_of_objective(self,normal_vector):
        func = lambda x: normal_vector  # minimize (negative sign)
        return func

    def get_hessian_of_objective(self):
        func = lambda x: self.hess_of_objective_cache_value
        return func

    def get_hessian_p_of_objective(self):
        func = lambda x,p: self.hessp
        return func

    def get_minimize_objective_function(self, x0, normal_vector):
        func = lambda x: -np.dot(x - x0, normal_vector)  # minimize (negative sign)
        return func

    def get_minimize_jacobian_of_objective(self,normal_vector):
        func = lambda x: -normal_vector  # minimize (negative sign)
        return func

    def get_SLSQP_constraints(self, x_original, x0, radius):
        '''
        :param x0: the center of a circle's point
        :param x_original: the original benign image
        :param radius: the radius of the circle
        :return:
        '''
        func_orthogonal = lambda x: np.dot(x - x0, x - x_original)
        func_radius = lambda x: np.sum(np.square(x - x0)) - np.square(radius)
        cons = ({"type": "eq", "fun": func_orthogonal, "jac": lambda x: 2 * x - x_original - x0},
                {"type": "eq", "fun": func_radius, "jac": lambda x: 2 * (x - x0)})  # jac是梯度/导数公式
        return cons


    def get_trust_constr_constraints(self, x_original, x0, radius):
        '''
        :param x0: the center of a circle's point
        :param x_original: the original benign image
        :param radius: the radius of the circle
        :return:
        '''
        func_orthogonal = lambda x: [np.dot(x - x0, x - x_original)]
        func_radius = lambda x: [np.sum(np.square(x - x0)) - np.square(radius)]
        cons_orthogonal = NonlinearConstraint(func_orthogonal, lb=0.0,ub=0.0,
                                              jac=lambda x: [2 * x - x_original - x0],
                                              hess=lambda x,v: self.hess_of_constraint_cache_value)
        cons_radius = NonlinearConstraint(func_radius, lb=0.0, ub=0.0,
                                              jac=lambda x: [2 * (x - x0)],
                                            hess=lambda x,v: self.hess_of_constraint_cache_value)
        cons = [cons_orthogonal, cons_radius]
        return cons


    def solve_tangent_point(self, x_original, x0, normal_vector, radius, max_iters):
        x = np.random.rand(x0.shape[0]).astype(np.float32)
        if self.initial_with_penalty_method:
            log.info("begin to use pytorch for minimizing")
            if isinstance(x_original,np.ndarray):
                x_original = torch.from_numpy(x_original).float().cuda()
            if isinstance(x0, np.ndarray):
                x0 = torch.from_numpy(x0).float().cuda()
            if isinstance(normal_vector, np.ndarray):
                normal_vector = torch.from_numpy(normal_vector).float().cuda()

            def get_objective_function(x, x0, normal_vector):
                loss = -torch.dot(x - x0, normal_vector)  # minimize (negative sign)
                return loss

            def get_constraints(x, x_original, x0, radius):
                '''
                :param x0:torch.tensor : the center of a circle's point
                :param x_original: the original benign image
                :param radius: the radius of the circle
                :return:
                '''
                loss_orthogonal = torch.dot(x - x0, x - x_original)
                loss_radius = torch.sum(torch.square(x - x0)) - np.square(radius).item()
                return loss_orthogonal, loss_radius
            x = torch.from_numpy(x).type(x0.dtype).cuda()
            x.requires_grad_()
            optimizer = SGD([x], lr=1e-3, momentum=0.95)
            for i in range(max_iters):
                objective_loss = get_objective_function(x, x0, normal_vector)
                contraint_orth, contraint_radius = get_constraints(x, x_original, x0, radius)
                loss = objective_loss + torch.square(contraint_orth) + torch.square(contraint_radius)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    x[:] = x.clamp(self.clip_min, self.clip_max)
            x = torch.clamp(x0 + radius * (x - x0) / torch.norm(x - x0, p=2), self.clip_min, self.clip_max)
            log.info("solve by using pytorch over!")
        if self.use_scipy:

            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if torch.is_tensor(x_original):
                x_original = x_original.detach().cpu().numpy()
            if torch.is_tensor(x0):
                x0 = x0.detach().cpu().numpy()
            if torch.is_tensor(normal_vector):
                normal_vector = normal_vector.detach().cpu().numpy()
            log.info("begin use {} to solve".format(self.scipy_method))
            if self.scipy_method=="SLSQP":
                result = minimize(self.get_minimize_objective_function(x0,normal_vector), x, jac=self.get_minimize_jacobian_of_objective(normal_vector),
                                  method='SLSQP', bounds=self.bounds,
                                  constraints=self.get_SLSQP_constraints(x_original, x0, radius),
                                  options={"disp": True})
            elif self.scipy_method=="trust-constr":
                result = minimize(self.get_minimize_objective_function(x0,normal_vector),x, method='trust-constr',
                                  jac=self.get_minimize_jacobian_of_objective(normal_vector),bounds=self.bounds,
                                  options={'verbose': 1},
                                  constraints=self.get_trust_constr_constraints(x_original, x0, radius),
                                  hess=self.get_hessian_of_objective(),hessp=self.get_hessian_p_of_objective())
            else:
                raise Exception("the scipy method must be one of SLSQP or trust-constr")
            log.info("use {} solve over!".format(self.scipy_method))
            return torch.from_numpy(result.x).float().cuda(), result.success

        return x,  True

