import torch
import numpy as np
import glog as log


def calculate_projection_of_x_original(x_original, x_of_plane, normal_vector):
    t = torch.dot(x_of_plane-x_original, normal_vector) / torch.sum(torch.square(normal_vector))
    projection_point = x_original + t * normal_vector
    return projection_point

def compute_line_plane_intersect_point(x1,x2,plane_point,plane_normal_vector):
    t = (torch.dot(plane_normal_vector, plane_point) - torch.dot(plane_normal_vector, x1))/ torch.dot(plane_normal_vector, x2-x1)
    return t * (x2-x1) + x1

class TangentFinder(object):
    def __init__(self, x, ball_center, radius, plane_normal_vector,  norm="l2"):
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
        return - torch.dot(self.ox, self.u) / (torch.norm(self.ox, p=self.ord) * torch.norm(self.u, p=self.ord))

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
        numerator = self.ox - torch.dot(self.ox, self.u) * self.u / torch.norm(self.u) ** 2
        ok_prime = (numerator / torch.norm(numerator, p=self.ord)) * self.R * self.cos_gamma()
        ok = ok_prime + self.get_height_of_K() * self.u / torch.norm(self.u)
        # log.info("[hemisphere] x_k is {:.4f} z_k is {:.4f}".format(self.R * self.cos_gamma(), self.get_height_of_K()))
        # assert self.R * self.cos_gamma() > 0, "cos(gamma) < 0! it equals to {}".format( self.R * self.cos_gamma())
        # assert self.get_height_of_K()  > 0, "h < 0! it equals to {}".format(self.get_height_of_K())
        # print("x_k is {}, z_k is {} in ball".format(self.R * self.cos_gamma(), self.get_height_of_K()))
        return ok + self.o


if __name__ == "__main__":
    r = 2.5
    import math
    import numpy as np
    all_tangent_point_list = []
    for angle in np.arange(-math.pi/2.0,math.pi/2.0,0.1):
        normal_vector = torch.zeros(3).double()
        normal_vector[2] = 1
        normal_vector[1] = 1 * math.tan(angle)
        normal_vector /= torch.norm(normal_vector)
        t = TangentFinder(torch.tensor([1.5*r,0,-0.5*r]).double(), torch.zeros(3).double(), r, normal_vector,"l2")
        print(tuple(t.compute_tangent_point().numpy().tolist()))
    print("_")
    normal_vector = torch.zeros(3).double()
    normal_vector[2] = -1
    normal_vector /= torch.norm(normal_vector)
    t = TangentFinder(torch.tensor([1.5 * r, 0, -0.5 * r]).double(), torch.zeros(3).double(), r, normal_vector, "l2")
    print(tuple(t.compute_tangent_point().numpy().tolist()))
    # #
    # normal_vector = torch.zeros(3)
    # normal_vector[2] = 1
    # t = TangentFinder(torch.tensor([1.5 * r, 0, -0.5 * r]), torch.zeros(3), r, normal_vector, "l2")
    # tangent = t.compute_tangent_point()
    # print(tangent)
    # # 计算直线和平面的交点
    # print(compute_line_plane_intersect_point(torch.tensor([1.5*r,0,-0.5*r]), tangent, torch.zeros(3),normal_vector))


    # figure 1
    normal_vector = torch.zeros(3).float()
    normal_vector[-1] = 1
    normal_vector /= torch.norm(normal_vector)
    # x = torch.tensor([2,2]).float()
    # t = TangentFinder(x, torch.tensor([4,3]).float(), 2, normal_vector, "l2")
    # tangent = t.compute_tangent_point()
    # print(tangent)
    # # 计算直线和平面的交点
    # print(compute_line_plane_intersect_point(x, torch.tensor([4,4]).float(), torch.tensor([4,3]).float(),
    #                                          normal_vector))
    # print(compute_line_plane_intersect_point(x, tangent,
    #                                          torch.tensor([4, 3]).float(),
    #                                          normal_vector))

    #
    r = 2.5
    x = torch.tensor([1.5*r,0,-0.5*r]).float()
    k = torch.tensor([2.1124, 0.0000, 1.3371]).float()

    print(compute_line_plane_intersect_point(x, k,
                                             torch.tensor([r, r,0]).float(),
                                             normal_vector))
    # t = TangentFinder(x, torch.zeros(3), r, normal_vector, "l2")
    # tangent = t.compute_tangent_point()
    # print(tangent)