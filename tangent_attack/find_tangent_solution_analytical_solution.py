import torch
import numpy as np

def calculate_projection_of_x_original(x_original, x_of_plane, normal_vector):
    t = torch.dot(x_of_plane-x_original, normal_vector) / torch.sum(torch.square(normal_vector))
    projection_point = x_original + t * normal_vector
    return projection_point

class TangentFinder(object):
    def __init__(self, x, ball_center, radius, plane_normal_vector, norm="l2"):
        '''
        :param x: the original image which is outside the ball, vector form.
        :param ball_center: the perturbed image lies on the decision boundary
        :param radius: the radius of the ball
        :param plane_normal_vector: the normal vector of hyperplane
        :param norm: l2 or linf
        '''
        self.x = x
        self.o = ball_center
        self.ox = self.x - self.o
        self.R = radius
        self.norm = norm
        self.u = plane_normal_vector # unit normal vector
        self.ord = np.inf if self.norm == "linf" else 2
        # assert self.sin_alpha() ** 2 + self.cos_alpha() ** 2 - 1.0 < 0.00001, "alpha assert error:{}".format(self.sin_alpha() ** 2 + self.cos_alpha() ** 2)
        # assert self.sin_beta() ** 2 + self.cos_beta() ** 2 - 1.0 < 0.00001, "beta assert error:{}".format(self.sin_beta() ** 2 + self.cos_beta() ** 2)
        # assert self.sin_gamma() ** 2 + self.cos_gamma() ** 2 - 1.0 < 0.00001, "gamma assert error:{}".format(self.sin_gamma() ** 2 + self.cos_gamma() ** 2)

    def sin_alpha(self):
        return - torch.dot(self.ox, self.u) / torch.norm(self.ox, p=self.ord)

    # def cos_alpha(self):
    #     return torch.sqrt(torch.square(torch.norm(self.ox, self.ord)) - torch.square(torch.dot(self.ox, self.u))) / torch.norm(self.ox, p=self.ord)
    def cos_alpha(self):
        return torch.sqrt(1- torch.square(self.sin_alpha()))

    def cos_beta(self):
        return self.R / torch.norm(self.ox, self.ord)

    # def sin_beta(self):
    #     return torch.sqrt(torch.square(torch.norm(self.ox, self.ord))  - self.R ** 2)/ torch.norm(self.ox, p=self.ord)
    def sin_beta(self):
        return torch.sqrt(1 - torch.square(self.cos_beta()))

    def sin_gamma(self):
        return self.sin_beta() * self.cos_alpha() - self.cos_beta() * self.sin_alpha()

    def cos_gamma(self):
        return self.cos_beta() * self.cos_alpha() + self.sin_beta() * self.sin_alpha()

    def get_height_of_K(self):
        return self.R * self.sin_gamma()

    def compute_tangent_point(self):
        numerator = self.ox - torch.dot(self.ox, self.u) * self.u
        ok_prime = (numerator / torch.norm(numerator, p=self.ord)) * self.R * self.cos_gamma()
        ok = ok_prime + self.get_height_of_K() * self.u
        return ok + self.o
