import random
import math
import numpy as np

def cal_x(y,y1,y2,x1,x2):
    return (y-y1)/(y2-y1) * (x2-x1) + x1

def sample_point_inside_circle(O, R, num):
    rho = R * np.sqrt(np.random.uniform(0, 1, num))
    phi = np.random.uniform(0, 2 * np.pi, num)

    x = O[0] + rho * np.cos(phi)
    y = O[1] + rho * np.sin(phi)

    return x,y


def sq_point_in_circle(O,r):
    """
    Generate a random point in an r radius circle
    centered around the start of the axis
    """

    t = 2*math.pi*np.random.uniform()
    R = (np.random.uniform(0,1) ** 0.5) * r

    return (O[0] + R*math.cos(t), O[1]  + R*math.sin(t))

def compute_line_plane_intersect_point(x1, x2, plane_point, plane_normal_vector):
    t = (np.dot(plane_normal_vector, plane_point) - np.dot(plane_normal_vector, x1))/ np.dot(plane_normal_vector, x2-x1)
    return t * (x2-x1) + x1

if __name__ == "__main__":
    r=2.0
    Ox = 1.5 * r
    Oy = 0
    num=15
    height = r
    k = np.array([1.6899, 0.0000, 1.0697])
    x = np.array([1.5*r,0,-0.5*r])
    r_on_height_fun = lambda y, x1,y1,x2,y2: x1 - ((y - y1)/(y2-y1)* (x2-x1) + x1)
    r_on_height = r_on_height_fun(height, x[0],x[2],k[0],k[2])
    print(r_on_height)

    x,y = sample_point_inside_circle([Ox,Oy], r_on_height, num)
    z = np.stack([x,y]).transpose((1,0))
    z = np.concatenate([z, np.ones(num).reshape(num,1) * height],axis=1)
    normal_vector = np.zeros(3)
    normal_vector[2] = 1
    normal_vector /= np.linalg.norm(normal_vector)
    z =[tuple(zz) for zz in z]
    print(z)
    x = np.array([1.5*r,0,-0.5*r])
    inter_set = []
    for z_i in z:
        inter_set.append(compute_line_plane_intersect_point(x, z_i, np.zeros(3), normal_vector))
    z = [tuple(zz.tolist()) for zz in inter_set]
    print(z)
    # r = 2.5
    # N = 20
    # points = np.array([sq_point_in_circle([Ox,Oy],r) for i in range(N)])

    x=np.array([2,2])
    HSJA = np.array([4,4])
    other_point = np.array([3.1339745962155616,3.5])
    O = np.array([4,3])
    normal_vector = np.array([0,1])
    joint_HSJA = compute_line_plane_intersect_point(x,HSJA,O,normal_vector)
    joint_other = compute_line_plane_intersect_point(x, other_point, O, normal_vector)
    print(joint_HSJA)
    print(joint_other)
