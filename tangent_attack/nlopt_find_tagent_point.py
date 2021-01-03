import nlopt
import numpy as np

def objective_function(x,grad, x0, normal_vector):
    if grad.size>0:
        grad[:] = normal_vector
    return np.dot(x-x0, normal_vector)

def constraint_orthogonal(x, grad, x_original, x0):
    if grad.size > 0:
        grad[:] = 2*x - x_original - x0
    return np.dot(x-x0, x-x_original)

def constraint_radius(x, grad, x0, radius):
    if grad.size > 0:
        grad[:] = 2*(x-x0)
    return np.sum(np.square(x-x0)) - np.square(radius)

def solve_tangent_point(x_original, x0, normal_vector, radius, clip_min=0,clip_max=1,max_iters=1000):
    initial_x = np.random.rand(x0.size)
    opt = nlopt.opt(nlopt.LD_SLSQP, x0.size)
    # local_opt = nlopt.opt(nlopt.LN_BOBYQA, x0.size)

    opt.set_xtol_rel(1e-3)
    opt.set_ftol_rel(1e-3)
    opt.set_ftol_abs(1e-4)
    # local_opt.set_xtol_rel(1e-3)
    # local_opt.set_ftol_rel(1e-3)
    # local_opt.set_ftol_abs(1e-4)

    opt.set_lower_bounds(np.ones(x0.size) * clip_min)
    opt.set_upper_bounds(np.ones(x0.size) * clip_max)
    # opt.set_local_optimizer(local_opt)
    opt.set_max_objective(lambda x,grad: objective_function(x, grad, x0, normal_vector))
    opt.add_equality_constraint(lambda x,grad: constraint_orthogonal(x, grad, x_original, x0),1e-8)
    opt.add_equality_constraint(lambda x,grad: constraint_radius(x, grad, x0, radius),1e-8)
    maximum_x = opt.optimize(initial_x)
    status_code = opt.last_optimize_result()
    return maximum_x, status_code > 0
