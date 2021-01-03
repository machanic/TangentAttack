import torch
from torch.optim import SGD, Adam, RMSprop, AdamW, Adagrad
import numpy as np
import glog as log

'''
using Augmented Lagrangian Method
'''

def get_objective_function(x, x0, normal_vector):
    loss = -torch.dot(x-x0, normal_vector)  # minimize (negative sign)
    return loss


def get_constraints(x, x_original, x0, radius):
    '''
    :param x0:torch.tensor : the center of a circle's point
    :param x_original: the original benign image
    :param radius: the radius of the circle
    :return:
    '''
    loss_orthogonal = torch.dot(x-x0, x-x_original)
    loss_radius = torch.sum(torch.square(x-x0)) - np.square(radius).item()
    return loss_orthogonal, loss_radius



def solve_tangent_point(x_original, x0, normal_vector, radius, clip_min=0.0, clip_max=1.0,max_iters=1000):
    # if x is None:
    x = torch.from_numpy(np.random.rand(x0.size(0))).type(x0.dtype).cuda()

    x.requires_grad_()

    # z = torch.from_numpy(np.random.rand(x0.size(0))).type(x0.dtype).cuda()
    # s = torch.from_numpy(np.random.rand(x0.size(0))).type(x0.dtype).cuda()
    # if x.numel() > 200:
    optimizer = SGD([x], lr=1e-3,momentum=0.95)
    # else:
    #     optimizer = SGD([x], lr=1e-3, momentum=0.95)
    lambd_orth = 1.0
    lambd_radius = 1.0
    rho_orth = 200.0
    rho_radius = 200.0

    # x <= 1 ==> x - 1 <= 0 ==> x - 1 + z^2 = 0
    # x >= 0 ==> x - s^2 = 0
    ones_upper = torch.ones_like(x) * clip_max
    rho_upper = 1.0
    rho_lower = 1.0
    lambd_upper = 1.0
    lambd_lower = 1.0
    for i in range(max_iters):
        objective_loss = get_objective_function(x, x0, normal_vector)
        # upper_bound_constraint_loss = x-ones_upper+torch.square(z)
        # lower_bound_constraint_loss = x-torch.square(s)
        contraint_orth, contraint_radius = get_constraints(x, x_original, x0, radius)
        # loss = objective_loss + 0.5 * (rho_orth * torch.square(contraint_orth) + rho_radius * torch.square(contraint_radius))\
        loss = objective_loss + torch.square(contraint_orth)  + torch.square(contraint_radius)
           # + lambd_orth * contraint_orth + lambd_radius * contraint_radius
              #  + 0.5 * (rho_upper * torch.square(upper_bound_constraint_loss) + rho_lower * torch.square(lower_bound_constraint_loss)) \
              # + lambd_upper * upper_bound_constraint_loss + lambd_lower * lower_bound_constraint_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lambd_orth = lambd_orth + rho_orth * contraint_orth.item()
        # lambd_radius = lambd_radius + rho_radius * contraint_radius.item()
        # rho_orth += 1
        # rho_radius += 1
        with torch.no_grad():
            x[:] = x.clamp(clip_min, clip_max)
    # orth_val = torch.dot(x - x0, x - x_original).item()
    # radius_diff_val = torch.sum(torch.square(x - x0)).item() - np.square(radius).item()
    # log.info("orthogonal inner product constraint: {:.4f}, radius square difference: {:.4f}".format(orth_val, radius_diff_val))
    return torch.clamp(x0 + radius * (x - x0)/ torch.norm(x-x0, p=2),clip_min,clip_max), True
    # return x.detach(), True