from scipy.optimize import minimize
import numpy as np
import glog as log

def get_objective_function(ball_center, normal_vector):
    func = lambda x: -np.dot(x-ball_center, normal_vector)  # minimize (negative sign)
    jac = lambda x: -normal_vector  # jac是梯度/导数公式
    return func, jac


def get_constraints(x_original, ball_center, radius):
    '''
    :param ball_center: the center of a circle's point
    :param x_original: the original benign image
    :param radius: the radius of the circle
    :return:
    '''
    func_orthogonal = lambda x: np.dot(x-ball_center, x-x_original)
    func_radius = lambda x: np.sum(np.square(x-ball_center)) - np.square(radius)
    cons = ({"type":"eq","fun": func_orthogonal,"jac":lambda x: 2*x - x_original - ball_center},
            {"type":"eq","fun": func_radius, "jac":lambda x: 2*(x-ball_center)})  # jac是梯度/导数公式
    return cons

def solve_tangent_point(x_original, ball_center, normal_vector, radius, clip_min=0.0, clip_max=1.0):
    assert isinstance(x_original,np.ndarray)
    initial_x = np.random.rand(ball_center.size)
    bounds = [(clip_min, clip_max) for _ in range(ball_center.size)]
    objective_func, objective_func_deriv = get_objective_function(ball_center, normal_vector)
    cons = get_constraints(x_original, ball_center, radius)
    result = minimize(objective_func, initial_x, jac=objective_func_deriv, method='SLSQP',bounds=bounds, constraints=cons,
                      options={"disp":True})
    log.info(result)
    return result.x
