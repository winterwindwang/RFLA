import os
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import torch


def get_data_iter(data_path, image_size=224, batch_size=50, dataset_name="imagenet"):
    datas = load_dataset(data_path, dataset_name)
    batch_num = len(datas) // batch_size
    for i in range(batch_num):
        batch_datas = datas[i * batch_size: (i + 1) * batch_size, :, :, :]
        imgs = []
        for img_pth in batch_datas:
            img = Image.open(img_pth).convert("RGB")
            img = ImageOps.fit(img, size=(image_size,image_size))
            img = np.array(img)
            imgs.append(img)
        yield np.array(imgs)


def get_image_iter_by_num(data_path, image_size=224, dataset_name="imagenet", test_num=100):
    datas = load_dataset(data_path, dataset_name)[:test_num]
    for i, img_pth in enumerate(datas):
        filename = os.path.basename(img_pth).split(".")[0]
        img = Image.open(img_pth).convert("RGB")
        img = ImageOps.fit(img, size=(image_size, image_size))
        img = np.array(img)
        yield np.array(img), filename

def get_image_iter(data_path, image_size=608, dataset_name="imagenet"):
    datas = load_dataset(data_path, dataset_name)
    for i, img_pth in enumerate(datas):
        filename = os.path.basename(img_pth).split(".")[0]
        img = Image.open(img_pth).convert("RGB")
        img = ImageOps.fit(img, size=(image_size, image_size))
        img = np.array(img)
        yield img, filename


def get_image_iter_OD(data_path, cover_images, image_size=608, dataset_name="imagenet"):
    datas = load_dataset(data_path, dataset_name)
    for i, img_pth in enumerate(datas):
        filename = os.path.basename(img_pth).split(".")[0]
        img = Image.open(img_pth).convert("RGB")
        img = ImageOps.fit(img, size=(image_size, image_size))
        img = np.array(img)
        cover_images = np.array(cover_images)
        yield img, cover_images, filename


def load_dataset(data_dir, dataset_name='imagenet'):
    if "imagenet" in dataset_name:
        image_list = glob(os.path.join(data_dir, "*.JPEG"))
        image_list += glob(os.path.join(data_dir, "*.png"))
        image_list += glob(os.path.join(data_dir, "*.jpg"))
    elif "traffic_sign" in dataset_name:
        image_list = glob(os.path.join(data_dir, "*.JPEG"))
        image_list += glob(os.path.join(data_dir, "*.png"))
        image_list += glob(os.path.join(data_dir, "*.jpg"))
    else:
        images = []
    return image_list


def get_test_dataloader(clean_path, adv_path, image_label_dict, image_size=224,batch_size=50, transform=None):
    file_list = os.listdir(adv_path)
    batch_num = len(file_list) // batch_size
    for i in range(batch_num):
        batch_datas = file_list[i * batch_size: (i + 1) * batch_size]
        imgs, adv_imgs, labels = [], [], []
        for filename in batch_datas:
            try:
                label = image_label_dict[filename.split(".")[0]]
            except:
                label = image_label_dict[filename.split("_")[0]]
            try:
                img = Image.open(os.path.join(clean_path, filename))
            except:
                img = Image.open(os.path.join(clean_path, f"{filename.split('_')[0]}.png"))

            adv_img = Image.open(os.path.join(adv_path, filename))
            img = ImageOps.fit(img, size=(image_size,image_size))
            adv_img = ImageOps.fit(adv_img, size=(image_size,image_size))
            img = transform(img).unsqueeze(dim=0)
            adv_img = transform(adv_img).unsqueeze(dim=0)

            imgs.append(img)
            adv_imgs.append(adv_img)

            labels.append(torch.LongTensor([label]))
        yield torch.cat(imgs, dim=0), torch.cat(adv_imgs, dim=0), torch.cat(labels)