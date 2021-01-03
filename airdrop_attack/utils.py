import torch

def calculate_projection_of_x_original(x_original, x_of_plane, normal_vector):
    t = (torch.dot(normal_vector, x_of_plane) - torch.dot(normal_vector, x_original)) / torch.sum(torch.square(normal_vector))
    projection_point = x_original + t * normal_vector
    return projection_point