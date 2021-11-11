import torch
import numpy as np
import math
import glog as log



class TangentFinder(object):
    def __init__(self, x, ball_center, short_radius, long_radius, plane_normal_vector, norm="l2"):
        '''
        :param x: the original image which is outside the ball, vector form.
        :param ball_center: the perturbed image lies on the decision boundary
        :param short_radius: the radius of the ball
        :param plane_normal_vector: the normal vector of hyperplane
        :param norm: l2 or linf
        '''
        self.x = x
        self.o = ball_center
        self.ox = self.x - self.o
        self.S = short_radius
        self.L = long_radius

        self.norm = norm
        self.u = plane_normal_vector # unit normal vector
        self.ord = np.inf if self.norm == "linf" else 2


        # assert self.sin_alpha() ** 2 + self.cos_alpha() ** 2 - 1.0 < 0.00001, "alpha assert error:{}".format(self.sin_alpha() ** 2 + self.cos_alpha() ** 2)
        # assert self.sin_beta() ** 2 + self.cos_beta() ** 2 - 1.0 < 0.00001, "beta assert error:{}".format(self.sin_beta() ** 2 + self.cos_beta() ** 2)
        # assert self.sin_gamma() ** 2 + self.cos_gamma() ** 2 - 1.0 < 0.00001, "gamma assert error:{}".format(self.sin_gamma() ** 2 + self.cos_gamma() ** 2)

    def compute_2D_x(self):
        x_norm = torch.norm(self.ox, p=self.ord)
        theta = self.theta()
        return (x_norm * torch.sin(theta), -x_norm * torch.cos(theta))

    def compute_tangent_point_of_ellipse(self):
        S = self.S
        L = self.L
        x_2D = self.compute_2D_x()
        x0 = x_2D[0].item()
        z0 = x_2D[1].item()
        in_sqrt = -L**2*S**2 + L**2*x0**2 + S**2*z0**2
        if in_sqrt < 0:
            log.info("x_k inside sqrt < 0 ({:.7f})! convert to 0.".format(in_sqrt))
            in_sqrt = 0
        if L**2*x0**2 + S**2*z0**2 < 1e-5:
            raise ZeroDivisionError("The denominator is zero!")
        # xk = S**2*(L**2 - z0*(L**2*S**2*z0/(L**2*x0**2 + S**2*z0**2) - L**2*x0*math.sqrt(in_sqrt)/(L**2*x0**2 + S**2*z0**2)))/(L**2*x0)
        # zk = L**2*S**2*z0/(L**2*x0**2 + S**2*z0**2) - L**2*x0*math.sqrt(in_sqrt)/(L**2*x0**2 + S**2*z0**2)
        # the below is orig
        #xk = S**2 * (L**2 - z0 * (L**2 * S**2 * z0/ (L**2 * x0**2 + S**2  * z0 ** 2) - L**2 * x0 * math.sqrt(-L**2 * S**2 + L**2 * x0 ** 2 + S**2 * z0 ** 2)/(L**2 * x0**2 + S**2 * z0**2))) / (L**2 * x0)
        #zk = (L**2 * S**2 * z0)/(L**2 * x0 ** 2 + S**2 * z0**2) - (L**2 * x0 * math.sqrt(-L**2 * S**2 + L**2  * x0**2 + S**2 * z0**2))/(L**2 * x0 ** 2 + S**2 * z0**2)
        xk = S**2 * (L**2 - z0 * (L**2 * S**2 * z0/ (L**2 * x0**2 + S**2 * z0**2) + L**2 * x0 * math.sqrt(-L**2 * S**2 + L**2 * x0 ** 2 + S**2 * z0 ** 2)/(L**2 * x0**2 + S**2 * z0**2))) / (L**2 * x0)
        zk = (L**2 * S**2 * z0)/(L**2 * x0 ** 2 + S**2 * z0**2) + (L**2 * x0 * math.sqrt(-L**2 * S**2 + L**2  * x0**2 + S**2 * z0**2))/(L**2 * x0 ** 2 + S**2 * z0**2)

        # result.append((xk,zk))
        # xk2 = S**2*(L**2 - z0*(L**2*S**2*z0/(L**2*x0**2 + S**2*z0**2) + L**2*x0*math.sqrt(-L**2*S**2 + L**2*x0**2 + S**2*z0**2)/(L**2*x0**2 + S**2*z0**2)))/(L**2*x0)
        # zk2 = L**2*S**2*z0/(L**2*x0**2 + S**2*z0**2) + L**2*x0*math.sqrt(-L**2*S**2 + L**2*x0**2 + S**2*z0**2)/(L**2*x0**2 + S**2*z0**2)
        # result.append((xk2, zk2))
        # assert zk > 0, zk
        return xk, zk

    def theta(self):
        return torch.acos(torch.dot(self.ox, -self.u)/(torch.norm(self.ox, p=self.ord) * torch.norm(self.u, p=self.ord)))

    def compute_tangent_point(self):
        x_k, z_k = self.compute_tangent_point_of_ellipse()
        numerator = self.ox - torch.dot(self.ox, self.u) * self.u / torch.norm(self.u) ** 2
        ok_prime = (numerator / torch.norm(numerator, p=self.ord)) * math.fabs(x_k)
        ok = ok_prime + z_k * self.u # / torch.norm(self.u)
        # print("x_k is {}, z_k is {} in ell".format(math.fabs(x_k), z_k))
        return ok + self.o


if __name__ == "__main__":
    sr = 1.0
    lr = 2.0
    import math
    import numpy as np

    x = torch.tensor([2,2]).float()
    all_tangent_point_list = []
    normal_vector = torch.zeros(2).double()
    normal_vector[0] = 0  # (0,1/tan(alpha),1)
    normal_vector[1] = 1
    normal_vector /= torch.norm(normal_vector)
    O = torch.zeros(2).double()
    O[0] = 4
    O[1] = 3
    t = TangentFinder(x, O, sr, lr, normal_vector, "l2")
    print(t.compute_tangent_point().numpy())
    #
    # #
    # normal_vector = torch.zeros(3)
    # normal_vector[2] = 1
    # t = TangentFinder(torch.tensor([1.5 * r, 0, -0.5 * r]), torch.zeros(3), r, normal_vector, "l2")
    # tangent = t.compute_tangent_point()
    # print(tangent)
    # # 计算直线和平面的交点
    # print(compute_line_plane_intersect_point(torch.tensor([1.5*r,0,-0.5*r]), tangent, torch.zeros(3),normal_vector))


    # figure 1
    # normal_vector = torch.zeros(3).float()
    # normal_vector[-1] = 1
    # normal_vector /= torch.norm(normal_vector)
    # x = torch.tensor([1,2]).float()
    # t = TangentFinder(x, torch.tensor([0,0,0]).float(),sr,lr, normal_vector, "l2")
    # tangent = t.compute_tangent_point()
    # print(tangent)
    # # 计算直线和平面的交点
    # print(compute_line_plane_intersect_point(x, torch.tensor([4,4]).float(), torch.tensor([4,3]).float(),
    #                                          normal_vector))
    # print(compute_line_plane_intersect_point(x, tangent,
    #                                          torch.tensor([4, 3]).float(),
    #                                          normal_vector))