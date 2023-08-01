import argparse
import random
import numpy as np
import torch
import torchvision
import yaml
from torchvision import models, transforms
import time
import data_loader
import utils_rfla 
from utils_rfla import *


class PSOAttack(object):
    def __init__(self,
                 c_list: list,
                 omega_bound: list,
                 lower_bound: list,
                 higher_bound: list,
                 v_lower: list,
                 v_higher: list,
                 model=None,
                 mask=None,
                 transform=None,
                 image_size=224,
                 dimension=8,
                 max_iter=10000,
                 size=50,
                 sub_size=50,
                 shape_type='tri',
                 save_dir='saved_images'
                 ):
        self.dimension = dimension  # the dimension of the base variable
        self.max_iter = max_iter  # maximum iterative number
        self.size = size  # the size of the particle
        self.sub_size = sub_size  # the number of the geometrical shapes in a circle
        self.pop_bound = []  # bound container
        self.pop_bound.append(lower_bound) # lower bound of solution
        self.pop_bound.append(higher_bound) # upper bound of solution
        self.omega_bound = omega_bound  # interia weight
        self.v_bound = []  # velocity container
        self.v_bound.append(v_lower)  # lower bound of velocity
        self.v_bound.append(v_higher)  # upper bound of velocity
        self.c_list = c_list  # learning factor
        self.mask = mask    # binary mask. ImageNet: all one matrix
        self.image_size = image_size  # image size
        self.shape_type = shape_type  # type of geometry shape 
        self.model = model  # the model
        self.transform = transform  # transformation for input of the model
        self.save_dir = save_dir

        if shape_type in ['line']:
            self.dimension += 0
        elif shape_type in ["triangle", "rectangle"]:
            self.dimension += 1
        elif shape_type in ["pentagon", "hexagon"]:
            self.dimension += 2

        self.pops = np.zeros((self.size, self.sub_size, self.dimension))  # Store all solutions
        self.v = np.zeros((self.size, self.sub_size, self.dimension))  # store all velocity
        self.p_best = np.zeros((self.size, self.dimension))  # the personal best solution
        self.g_best = np.zeros((1, self.dimension))[0]  # the global best solution in terms of sum of a circle
        self.sg_best = np.zeros((1, self.dimension))[0]  # the global best solution

        self.g_best_fitness = 0    # store the best fitness in terms of sum of a circle
        self.sg_best_fitness = 0   # store the best fitness score
        self.p_best_fitness = [0] * self.size  # store the person best fitness

    def set_model(self, model):
        self.model = model

    def set_mask(self, mask):
        self.mask = mask

    def check_circle_in_mask(self, point_x, point_y):
        if not self.mask[point_x, point_y]:
            return False
        return True

    def get_circle_raidus(self):
        """
        Random generate a circle with point (x, y) and raidus
        :return:
        """
        radius = np.random.uniform(self.pop_bound[0][2], self.pop_bound[1][2])
        point_x = np.random.randint(radius, self.image_size - radius)
        point_y = np.random.randint(radius, self.image_size - radius)
        while not self.check_circle_in_mask(point_x, point_y):
            point_x = np.random.randint(radius, self.image_size - radius)
            point_y = np.random.randint(radius, self.image_size - radius)
        return (point_x, point_y, radius)

    def initial_per_circle(self, circle):
        """
        generate solution for a circle
        :param circle:
        :return:
        """
        x, y, r = circle
        pops = []
        # r, x, y, alpha, red, green, blue, angle, angle
        for j in range(self.sub_size):  # 50个圆中的三角形
            alpha = np.random.uniform(self.pop_bound[0][3], self.pop_bound[1][3])
            red = np.random.randint(self.pop_bound[0][4], self.pop_bound[1][4])
            green = np.random.randint(self.pop_bound[0][5], self.pop_bound[1][5])
            blue = np.random.randint(self.pop_bound[0][5], self.pop_bound[1][5])
            angle = np.random.uniform(self.pop_bound[0][6], self.pop_bound[1][6])
            if self.shape_type in 'line':
                pops.append((x, y, r, alpha, red, green, blue, angle))
            elif self.shape_type in "triangle":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta))
            elif self.shape_type in "rectangle":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta))
            elif self.shape_type in "pentagon":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                _, angle_beta1 = utils_rfla.get_new_angle((x, y), r, [angle, angle_beta])  # 
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta, angle_beta1))
            elif self.shape_type in "hexagon":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                _, angle_beta1 = utils_rfla.get_new_angle((x, y), r, [angle, angle_beta])  # 
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta, angle_beta1))
            else:
                raise ValueError("Please select the shape in [line, triangle, rectangle, pentagon, hexagon]")
        return pops

    def gen_adv_images_by_pops(self, image, pops):
        result_images = []
        for j, pop in enumerate(pops):
            image_raw = copy.deepcopy(image)
            x, y, r, alpha, red, green, blue = pop[:7]
            angles = pop[7:]
            x_0, y_0 = utils_rfla.get_point_by_angle((x, y), r, angles[0])

            if self.shape_type in 'line':
                x_1, y_1 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                points = np.array([(x_0, y_0), (x_1, y_1)]).astype(np.int32)
                cv2.line(image_raw, points[0], points[1], color=(red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "triangle":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r,
                                                         angle_beta)
                x_11, y_11 = get_symmetric_point_of_center((x, y), (x_0, y_0))
                points = np.array([(x_0, y_0), (x_1, y_1), (x_11, y_11)]).astype(np.int32)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "rectangle":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta) 
                x_11, y_11 = get_symmetric_point_of_center((x, y), (x_0, y_0))
                x_21, y_21 = get_symmetric_point_of_center((x, y), (x_1, y_1))
                points = np.array([(x_0, y_0), (x_1, y_1), (x_11, y_11), (x_21, y_21)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "pentagon":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta) 

                angle_beta1 = angles[2]
                x_2, y_2 = utils_rfla.get_point_by_angle((x, y), r, angle_beta1) 

                x_21, y_21 = utils_rfla.get_symmetric_point_of_center((x, y), (x_2, y_2))
                x_31, y_31 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))

                points = np.array([(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_21, y_21), (x_31, y_31)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)

                cv2.polylines(image_raw, [points], True, (red, green, blue), 1)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "hexagon":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta)

                angle_beta1 = angles[2]  
                x_2, y_2 = utils_rfla.get_point_by_angle((x, y), r, angle_beta1)

                x_21, y_21 = utils_rfla.get_symmetric_point_of_center((x, y), (x_1, y_1))
                x_31, y_31 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                x_41, y_41 = utils_rfla.get_symmetric_point_of_center((x, y), (x_2, y_2))

                points = np.array(
                    [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_31, y_31), (x_21, y_21), (x_41, y_41)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)

                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)
            # Image.fromarray(image_new).save("optimizing_images/image_{j}.png")
            result_images.append(image_new)
        return np.array(result_images)

    def to_tensor(self, images):
        try:
            images = np.transpose(images, (0, 3, 1, 2))
        except:
            images = np.transpose(images, (2, 0, 1))
            images = np.expand_dims(images, axis=0)
        if self.transform is not None:
            # Covert the image to PIL.Image
            images = torch.cat([self.transform(Image.fromarray(img)).unsqueeze(dim=0) for img in images])
        else:
            images = images.astype(np.float32) / 255.
            images = torch.from_numpy(images)
        return images

    def initialize(self, image, label, filename="test"):
        image_raw = copy.deepcopy(image)
        temp = 1e+5
        temp_real = 1e+5
        for i in range(self.size): 
            x, y, r = self.get_circle_raidus()
            pops = self.initial_per_circle((x, y, r))

            circlr_v = [random.uniform(self.v_bound[0][k], self.v_bound[1][k]) for k in range(self.dimension) if k < 3]
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k < 3:
                        self.v[i][j][:3] = circlr_v
                        continue
                    self.v[i][j][k] = random.uniform(self.v_bound[0][k], self.v_bound[1][k])

            adv_images = self.gen_adv_images_by_pops(image_raw, pops)
            softmax = self.calculate_fitness(adv_images)
            fitness_score, indices = torch.max(softmax, dim=1)

            # Exit when the best solution is obtained
            success_indicator = (indices != label)
            if success_indicator.sum().item() >= 1:
                # adv_images[(indices != label).cpu().data.numpy()]
                image2saved = adv_images[success_indicator.cpu().data.numpy()]
                Image.fromarray(image2saved[0]).save(fr'{self.save_dir}/{filename}.png')
                print(
                    f"Initialize Success: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {indices[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                return True
            g_fitness = torch.sum(fitness_score).item()
            p_best_idx = torch.argmin(fitness_score)
            g_fitness_real = torch.min(fitness_score)

            self.pops[i, ...] = pops
            self.p_best[i] = pops[p_best_idx.item()]
            self.p_best_fitness[i] = fitness_score[p_best_idx.item()]

            if g_fitness < temp:
                self.g_best = self.p_best[i]
                self.g_best_fitness = g_fitness
                temp = g_fitness

            if g_fitness_real < temp_real:
                self.sg_best = self.p_best[i]
                self.sg_best_fitness = g_fitness_real
                temp_real = g_fitness_real
            print(
                f"Initialize Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness},  p_fitness: {self.p_best_fitness[i]}, prediction: {indices[0]}, probability: {fitness_score[0]}")
        return False

    @torch.no_grad()
    def calculate_fitness(self, images):
        images = self.to_tensor(images)
        output = self.model(images.cuda())
        softmax = torch.softmax(output, dim=1)
        return softmax

    def calculate_omega(self, itr):
        omega = self.omega_bound[1] - (self.omega_bound[1] - self.omega_bound[0]) * (itr / self.max_iter)
        return omega

    def update(self, image, label, itr, filename="test_update"):
        c1 = c_list[0]
        c2 = c_list[1]
        c3 = c_list[2]
        w = self.calculate_omega(itr)
        image_raw = copy.deepcopy(image)
        for i in range(self.size): 

            ################# Constrain the bound of the circle #################
            circlr_v = w * self.v[i][0, :3] + c1 * random.uniform(0, 1) * (
                    self.p_best[i][:3] - self.pops[i][0, :3]) + c2 * random.uniform(
                0, 1) * (self.g_best[:3] - self.pops[i][0, :3]) + c3 * random.uniform(
                0, 1) * (self.sg_best[:3] - self.pops[i][0, :3])

            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k > 3:
                        break
                    self.v[i][j][k] = min(max(self.v[i][j][k], self.v_bound[0][k]), self.v_bound[1][k])

            # 更新位置
            self.pops[i][:3] = self.pops[i][:3] + self.v[i][:3]
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k > 3:
                        break
                    self.pops[i][j][k] = min(max(self.pops[i][j][k], self.pop_bound[0][k]), self.pop_bound[1][k])

            ################# Finished #################

            ################# Constrain the bound of the reset variables #################
            for j in range(self.sub_size):
                self.v[i][j][:3] = circlr_v
                self.v[i][j][3:] = w * self.v[i][j][3:] + c1 * random.uniform(0, 1) * (
                        self.p_best[i][3:] - self.pops[i][j][3:]) + c2 * random.uniform(
                    0, 1) * (self.g_best[3:] - self.pops[i][j][3:]) + c3 * random.uniform(
                    0, 1) * (self.sg_best[3:] - self.pops[i][j][3:])
            # velocity bound
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    self.v[i][j][k] = min(max(self.v[i][j][k], self.v_bound[0][k]), self.v_bound[1][k])
            # update the solution
            self.pops[i] = self.pops[i] + self.v[i]
            # solution bound
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    self.pops[i][j][k] = min(max(self.pops[i][j][k], self.pop_bound[0][k]), self.pop_bound[1][k])
            ################# Finished!! #################

            current_adv_images = self.gen_adv_images_by_pops(image_raw, self.pops[i])
            softmax = self.calculate_fitness(current_adv_images)
            fitness_score, indices = torch.max(softmax, dim=1)

            # If find the best solution, then exist
            success_indicator = (indices != label)
            if success_indicator.sum().item() >= 1:
                image2saved = current_adv_images[success_indicator.cpu().data.numpy()]
                Image.fromarray(image2saved[0]).save(fr'{self.save_dir}/{filename}.png')
                print(
                    f"【{itr}/{self.max_iter}】Success: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {indices[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                return True

            g_fitness = torch.sum(fitness_score).item()
            p_best_idx = torch.argmin(fitness_score)
            g_fitness_real = torch.min(fitness_score)

            if fitness_score[p_best_idx.item()] < self.p_best_fitness[i]:
                self.p_best[i] = self.pops[i][p_best_idx.item()]

            if g_fitness < self.g_best_fitness:
                self.g_best_fitness = g_fitness
                self.g_best = self.p_best[i]

            if g_fitness_real < self.sg_best_fitness:
                self.sg_best_fitness = g_fitness_real
                self.sg_best = self.p_best[i]

            if itr == self.max_iter-1 and i == self.size-1:
                Image.fromarray(current_adv_images[0]).save(fr'{self.save_dir}/{filename}_failed.png')

            print(
                f"【{itr}/{self.max_iter}】Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {indices[0]}, probability: {fitness_score[0]}")

        return False

    def run_pso(self, data_loader):
        total = 0
        success_cnt = 0
        for i, (image, filename) in enumerate(data_loader):  # iterative each images
            total += 1
            softmax_ori = self.calculate_fitness(image)
            fitness_score_ori, pred_label = torch.max(softmax_ori, dim=1)
            print(f"filename: {filename}.png predicted as: {pred_label.item()}")
            is_find_init = self.initialize(image, pred_label, filename)
            if is_find_init:  
                success_cnt += 1
                print("Initial found!!!")
                print("==" * 30)
                continue

            for itr in range(self.max_iter):  
                is_find_search = self.update(image, pred_label, itr, filename)
                if is_find_search:  
                    success_cnt += 1
                    print("==" * 30)
                    break

        asr = round(100 * (success_cnt / total), 2)
        return asr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random Search Parameters")
    ################# the file path of config.yml #################
    parser.add_argument("--yaml_file", type=str, default="config.yml", help="the settings config")
    ################# load config.yml file    ##################
    known_args, remaining = parser.parse_known_args()
    with open(known_args.yaml_file, 'r', encoding="utf-8") as fr:
        yaml_file = yaml.safe_load(fr)
        parser.set_defaults(**yaml_file)
    ################# assign other file in bash #################
    parser.add_argument("--save_dir", type=str, default="saved_images")
    parser.add_argument("--model_name", type=str, default="resnet50", help="target model name")
    parser.add_argument("--shape_type", type=str, default="hexagon", help="line triangle rectangle pentagon hexagon")
    args = parser.parse_args(remaining)
    print(args)

    if "win" in args.sys_flag:
        # image_path = "../ImageNet-NIPS2017/images"
        image_path = "test_images"
    elif "linux" in args.sys_flag:
        image_path = '../NIPS2017-ImageNet1K/images'

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = f"saved_images/EXP_NAME_{args.shape_type}_{args.model_name}_{args.dataset_name}_{time_str}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask = np.ones((224, 224), dtype=np.uint8)
    image_loader = data_loader.get_image_iter(image_path, image_size=args.image_size, dataset_name=args.dataset_name)

    model = get_model(args.model_name, device)
    transform = transforms.ToTensor()
    

    # r, x, y, alpha, red, green, blue, angle, angle

    if "line" in args.shape_type:
        args.alpha_bound = [0, 1]
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0]]  
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size, args.alpha_bound[1], args.color_bound[1],
                        args.color_bound[1], args.color_bound[1], args.angle_boud[1]] 
    elif "rectangle" in args.shape_type or "triangle" in args.shape_type:
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0], args.angle_boud[0]] 
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size * 0.4, args.alpha_bound[1],
                        args.color_bound[1], args.color_bound[1], args.color_bound[1], args.angle_boud[1],
                        args.angle_boud[1]]  
    else:
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0], args.angle_boud[0],
                       args.angle_boud[0]]  
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size * 0.4, args.alpha_bound[1],
                        args.color_bound[1], args.color_bound[1], args.color_bound[1], args.angle_boud[1],
                        args.angle_boud[1], args.angle_boud[1]] 



    v_higher = np.array([5, 5, 10, 0.05, 5, 5, 5, 10, 10, 10])
    v_lower = -np.array(v_higher)

    c_list = [args.c1, args.c2, args.c3]
    pso = PSOAttack(mask=mask,
                    model=model,
                    image_size=args.image_size,
                    dimension=args.dimension,
                    max_iter=args.max_iter,
                    size=args.pop_size,
                    sub_size=args.sub_size,
                    c_list=c_list,
                    omega_bound=args.omega_bound,
                    lower_bound=lower_bound,
                    higher_bound=higher_bound,
                    v_lower=v_lower,
                    v_higher=v_higher,
                    shape_type=args.shape_type,
                    save_dir=save_dir)  

    asr = pso.run_pso(image_loader)
    print(f"ASR of {args.dataset_name}_{args.model_name} is: {asr}")
    print("Finished !!!")
