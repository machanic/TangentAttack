import torch
import math

class FakeNormalVector(object):
    def __init__(self, original_image, boundary_image, model, clip_min=0.0, clip_max=1.0, binary_search_threshold=1e-2):
        self.model = model
        self.batch = original_image.size(0)
        self.channels = original_image.size(1)
        self.height = original_image.size(2)
        self.width = original_image.size(3)
        self.x = original_image.view(-1)
        self.o = boundary_image.view(-1)
        self.binary_search_angle_threshold = binary_search_threshold
        self.radius = 1e-1
        self.p = None
        self.p_prime = None
        self.clip_min = clip_min
        self.clip_max = clip_max

    def calculate_projection_of_plane(self, x, x_of_plane, normal_vector):
        t = torch.dot(x_of_plane - x, normal_vector) / torch.norm(normal_vector)
        projection_point = x + t * normal_vector
        return projection_point

    def get_fake_hyperplane(self):
        return self.x - self.o, self.o # normal_vector, x_of_point

    def get_point_wrt_angle(self, angle, radius):
        main_plane_normal_vector, main_plane_point = self.get_fake_hyperplane()
        if self.p_prime is None:
            p_prime = torch.rand_like(self.x)
            p_prime = self.calculate_projection_of_plane(p_prime, main_plane_point, main_plane_normal_vector)
            self.p_prime = p_prime
        if self.p is None:
            p = torch.rand_like(self.x)
            p = self.calculate_projection_of_plane(p, self.p_prime, self.o - self.p_prime)
            p = self.calculate_projection_of_plane(p, main_plane_point, main_plane_normal_vector)
            self.p = p
        final_point = self.o + (self.p_prime - self.o) / torch.norm(self.p_prime - self.o,p=2) * radius * math.cos(angle) \
         + (self.p - self.p_prime) / torch.norm(self.p- self.p_prime,p=2) * radius * math.sin(angle)
        return final_point

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images.view(self.batch,self.channels,self.height,self.width))
        if target_labels is None:
            return logits.max(1)[1].detach().cpu() != true_labels
        else:
            return logits.max(1)[1].detach().cpu() == target_labels

    def binary_search_angle(self, true_labels, target_labels):
        num_evals = 0
        # Compute distance between each of perturbed image and original image.
        low = 0
        high = math.pi
        # Call recursive function.
        while (high - low) > self.binary_search_angle_threshold:
            # log.info("max in binary search func: {}, highs:{}, lows:{}, highs-lows: {} , threshold {}, (highs - lows) / thresholds: {}".format(torch.max((highs - lows) / thresholds).item(),highs, lows, highs-lows, thresholds, (highs - lows) / thresholds))
            # projection to mids.
            mid = (high + low) / 2.0
            mid_image = self.get_point_wrt_angle(mid, self.radius)
            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_image, true_labels, target_labels)
            num_evals += 1
            decisions = decisions.int()
            # 攻击成功时候high用mid，攻击失败的时候low用mid
            low = torch.where(decisions == 0, mid, low)  # lows:攻击失败的用mid，攻击成功的用low
            high = torch.where(decisions == 1, mid, high)  # highs: 攻击成功的用mid，攻击失败的用high, 不理解的可以去看论文Algorithm 1
            # log.info("decision: {} low: {}, high: {}".format(decisions.detach().cpu().numpy(),lows.detach().cpu().numpy(), highs.detach().cpu().numpy()))
        p_on_hyperplane = self.get_point_wrt_angle(high, self.radius)  # high表示classification boundary偏攻击成功一点的线
        return p_on_hyperplane, num_evals

    def compute_fake_normal_vector(self, true_labels, target_labels):
        # then sample points f
        p_on_hyperplane, num_evals = self.binary_search_angle(true_labels, target_labels)
        fake_normal_vector = torch.cross(self.x-self.o, p_on_hyperplane-self.o)
        fake_normal_vector = fake_normal_vector/ torch.norm(fake_normal_vector,p=2)
        # still needs to judge the direction of fake normal vector
        test_point = self.o + fake_normal_vector * self.radius * 2
        decisions = self.decision_function(test_point, true_labels, target_labels)
        decisions = decisions.int()
        if decisions[0].item() == 0:
            fake_normal_vector = -fake_normal_vector
        return fake_normal_vector