import math
import torch
import random
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def delete_square(inp_img, add_h, add_v, use_randomly, noise_pixels=1):
    """Delete random square from image for robustness to noise"""

    h, w = np.shape(inp_img)

    # Random starting pixel
    rh = random.randint(0, h)
    rw = random.randint(0, w)
    sub = noise_pixels

    # Boundries for square
    hmin = int(max(rh - sub, 0))
    hmax = int(min(rh + add_h, h - 1))
    vmin = max(rw - sub, 0)
    vmax = int(min(rw + add_v, w - 1))

    # randomly add either salt/pepper
    if use_randomly:
        salt = random.choice([True, False])
    else:
        salt = True
    if salt:
        inp_img[hmin:hmax, vmin:vmax] = np.full(
            shape=(hmax - hmin, vmax - vmin), fill_value=255
        )
    else:
        inp_img[hmin:hmax, vmin:vmax] = np.full(
            shape=(hmax - hmin, vmax - vmin), fill_value=0
        )
    inp_img = Image.fromarray(inp_img)

    return inp_img


def img_tranforms(transforms_cfg, image):
    for operator in transforms_cfg:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if op_name == "ColorJitter":
            for key, value in param.items():
                if type(value) == list:
                    min = value[0]
                    max = value[1]
                if key == "brightness":
                    assert (
                        min > 0 and max > 0
                    ), "brightness values must be Non Negative number"
                    transform = transforms.ColorJitter(brightness=(min, max))
                    image = transform(image)
                if key == "contrast":
                    assert (
                        min > 0 and max > 0
                    ), "contrast values must be Non Negative number"
                    transform = transforms.ColorJitter(contrast=(min, max))
                    image = transform(image)
                if key == "saturation":
                    assert (
                        min > 0 and max > 0
                    ), "saturation values must be Non Negative number"
                    transform = transforms.ColorJitter(saturation=(min, max))
                    image = transform(image)
                if key == "hue":
                    assert (
                        min >= -0.5 and max <= 0.5
                    ), "hue values must be in range [-0.5, 0.5]"
                    transform = transforms.ColorJitter(hue=(min, max))
                    image = transform(image)

        elif op_name == "Noise":
            use_randomly = False
            assert (
                param["type"] == "salt&pepper"
            ), "only salt&pepper (blocks) noise is available at the moment"
            for key, value in param.items():
                if key == "max_amount":
                    maximum_noise_amount = value
                elif key == "noise_block_height":
                    noise_block_height = value
                elif key == "noise_block_width":
                    noise_block_width = value
                elif key == "random" and value:
                    use_randomly = True
            cv_image = np.array(image)
            if use_randomly:
                if random.choice([True, False]):
                    noise_amount = random.randint(0, maximum_noise_amount)
                    for i in range(1, noise_amount + 1):
                        image = delete_square(
                            cv_image,
                            noise_block_height,
                            noise_block_width,
                            use_randomly,
                        )
            else:
                for i in range(1, maximum_noise_amount + 1):
                    image = delete_square(
                        cv_image, noise_block_height, noise_block_width, use_randomly
                    )
    return image


class RecognitionDataset(Dataset):
    def __init__(self, dataframe, client, cfg):
        self.df = dataframe
        self.cfg = cfg
        self.client = client
        self.cell_ids_list = self.df[self.df.columns[0]].values.tolist()
        self.labels_list = self.df["labels"].values.tolist()
        self.file_name_list = self.df["file_name"].values.tolist()

    def __len__(self):
        return len(self.cell_ids_list)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        cell_id = self.cell_ids_list[index]

        label = self.labels_list[index]
        file_name = self.file_name_list[index]
        resp = get_image(file_name, cell_id, self.client) # modified
        image = resp.convert("L")
        transforms_cfg = self.cfg["transforms"]
        if len(transforms_cfg) > 0:
            img_tranforms(transforms_cfg, image)

        return image, label


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            )

        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=64, imgW=200, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))
            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                resized_image = cv2.resize(
                    np.array(image),
                    (resized_w, self.imgH),
                    interpolation=cv2.INTER_CUBIC,
                )
                resized_images.append(transform(resized_image))
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels
