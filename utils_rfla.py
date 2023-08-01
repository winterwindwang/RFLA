import copy
import cv2
import numpy as np
import os
from glob import glob
from PIL import Image, ImageOps
import torch
from torchvision import models



def get_model(model_name, device):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    if "resnet50" in model_name.lower():
        model = models.resnet50(pretrained=True)
    if "resnet152" in model_name.lower():
        model = models.resnet152(pretrained=True)
    if "alexnet" in model_name.lower():
        model = models.alexnet(pretrained=True)
    elif "vgg16" in model_name.lower():
        model = models.vgg16(pretrained=True)
    elif "densenet121" in model_name.lower():
        model = models.densenet121(pretrained=True)
    elif "resnext" in model_name.lower():
        model = models.resnext50_32x4d(pretrained=True)
    elif "wideresnet" in model_name.lower():
        model = models.wide_resnet50_2(pretrained=True)
    elif "model_nameasnet" in model_name.lower():
        model = models.model_nameasnet1_0(pretrained=True)
    elif "squeezenet" in model_name.lower():
        model = models.squeezenet1_0(pretrained=True)

    model = torch.nn.Sequential(normalize, model)
    model.eval()
    model.to(device)
    return model


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """
    Differentiable version of functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def pixel_ell0_norm(image1, image2):
    diff = torch.abs(image1 - image2)
    diff = torch.sum(diff, dim=1)
    diff = torch.flatten(diff)
    return diff[diff > 0].shape[0]



def sort_points_by_distance(points):
    """
    利用最近点距对点进行排序
    :param points:
    :return:
    """
    points_res = [points[0]]
    cur_pt = points[0]
    points = np.delete(points, 0, axis=0)

    while len(points)!=0:
        min_dist = np.inf
        min_pt = None
        min_idx = 0
        for i, pt in enumerate(points):
            dist = np.sqrt((cur_pt[0] - pt[0])**2 + (cur_pt[1] - pt[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_pt = pt
                min_idx = i
        cur_pt = min_pt
        points_res.append(min_pt)
        points = np.delete(points, min_idx, axis=0)
    return np.array(points_res)


def get_new_angle(point_center, radius, angles):
    if not isinstance(angles, list):
        angles = [angles]
    new_angle = np.random.uniform(0, 360)

    def just_angles(angle, angles):
        for ang in angles:
            if abs(angle - ang) < 20:
                return False
            else:
                return True

    while not just_angles(new_angle, angles):
        new_angle = int(np.random.uniform(0, 360))

    new_point = get_point_by_angle(point_center, radius, new_angle)
    return new_point, new_angle


def get_point_by_angle(point, radius, angle):
    point_x, point_y = point
    return (point_x + radius * np.cos(angle * np.pi / 180), point_y + radius * np.sin(angle * np.pi / 180))


def get_point_in_circle_boundary(point: list, radius):
    """ Random get point from the circle boundary
    :param c_point: [1 x 2] data array, with (x,y) of the circle
    :param radius: the radius of the circle
    :return:
    """
    angle = int(np.random.uniform(-180, 180))
    point_canditate = get_point_by_angle(point, radius, angle)
    return point_canditate, angle


def calculate_cross_point_of_two_line(line1, line2):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    if (x12 - x11) == 0:  # L1直线的斜率不存在，即垂直于x轴
        k1 = None
        b1 = 0
    else:
        k1 = (y12 - y11) / (x12 - x11)
        b1 = y11 - k1 * x11

    if (x21 - x22) == 0:
        k2 = None
        b1 = 0
    else:
        k2 = (y22 - y21) / (x22 - x21)
        b2 = y21 - k2 * x21

    if k1 is None and k2 is None:
        if x11 == x21:  # 两条直线为同一条直线
            return [x11, y11]  # 返回任意一点
        else:
            # return None  # 平行线，无交点
            raise ValueError("two line are parallel!!")
    elif k1 is not None and k2 is None:  # line2垂直于x轴
        cross_point_x = x21
        cross_point_y = k1 * cross_point_x + b1
    elif k1 is None and k2 is not None:  # line1垂直于x轴
        cross_point_x = x11
        cross_point_y = k2 * cross_point_x + b2
    else:
        if k1 == k2:
            if b1 == b2:  # 两条直线为同一条直线，返回任意一点
                return [x11, y11]  # 返回任意一点
            else:
                raise ValueError("slope should not be equal!!")
        else:
            cross_point_x = (b2 - b1) * 1.0 / (k1 - k2)
            cross_point_y = k1 * cross_point_x + b1
    return (cross_point_x, cross_point_y)


def get_symmetric_point_of_center(point_center, point):
    """
    :param point_center: the center of the circle
    :param point:  other point in the bound of circle
    :return:
    """
    x, y = point_center
    x1, y1 = point
    x_res = 2 * x - x1
    y_res = 2 * y - y1
    return (x_res, y_res)


def calculate_cross_point_of_line_circle(line, circle):
    """
    :param line: a vector of two point [x1,y1,x2,y2]
    :param circle: a vector of [x,y,r] the center point and radius of the circle  (x-a)^2 + (y-b)^2 = r^2
    :return:
    """
    x1, y1, x2, y2 = line
    x, y, r = circle

    if x1 - x2 == 0:
        k = None
        b = 0
    else:
        k = (y2 - y1) * 1.0 / (x2 - x1)
        b = y1 - k * x1
    # calculate the distance between the center point of circle and the line
    if k is None:
        dist = abs(x - x1)
    else:
        dist = (k * x - y + b) / (k ** 2 + 1)
    if dist > r:
        raise ValueError("No cross point between line and circle")
        # return None
    if k is None:
        cross_point_x1 = x1
        cross_point_x2 = x1
        cross_point_y1 = np.sqrt(r ** 2 - (cross_point_x1 - x) ** 2) + y
        cross_point_y2 = -np.sqrt(r ** 2 - (cross_point_x2 - x) ** 2) + y
    else:
        A = (k ** 2 + 1)
        B = (2 * k * b - 2 * k * y - 2 * x)
        C = x ** 2 + y ** 2 - r ** 2
        delta = B ** 2 - 4 * A * C
        if delta > 0:  # there has two solution
            cross_point_x1 = (-B + np.sqrt(delta)) / (2 * A)
            cross_point_x2 = (-B - np.sqrt(delta)) / (2 * A)

            cross_point_y1 = k * cross_point_x1 + b
            cross_point_y2 = k * cross_point_x2 + b
        elif delta == 0:
            cross_point_x1 = - B / (2 ** A)
            cross_point_y1 = k * cross_point_x1 + b
            cross_point_x2 = None
            cross_point_y2 = None
        else:
            raise ValueError("There has no solution")
    return (cross_point_x1, cross_point_y1), (cross_point_x2, cross_point_y2)


def plot_circle(image, circle, color):
    x, y, r, alpha = circle
    image_raw = copy.deepcopy(image)
    cv2.circle(image_raw, (x, y), r, color, -1)
    image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)
