from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import cv2
import pickle
import random

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, CoarseDropout, ElasticTransform, Rotate, RandomCrop, Resize
)


def convert_uint_sensor(sens_array):
    convereted_arr = []
    for idx, sens in enumerate(sens_array):
        if sens > 128:
            print(sens)
            s = bin(int(sens)).replace("0b", "")
            num = '0' + s[1:]
            b = -1 * BitArray(bin=num).int
        else:
            b = sens
        convereted_arr.append(b)
    return np.array(convereted_arr)


def strong_aug(p=0.6, im_height=700, im_width=1200):
    dropout_w = int(im_width / 82)
    dropout_h = int(im_height / 9.)
    return Compose([
        Rotate(limit=2, p=0.5),
        RandomCrop(height=int(im_height * 0.95), width=int(im_width * 0.9), p=0.3),
        ElasticTransform(p=0.8),
        HorizontalFlip(p=0.5),
        CoarseDropout(max_holes=8, max_height=dropout_w, max_width=dropout_h, min_holes=1,
                      min_height=5, min_width=5, fill_value=0, always_apply=False, p=0.85),
        OneOf([
            MotionBlur(p=0.8),
            Blur(blur_limit=20, p=0.8),
        ], p=0.35),

        Resize(height=256, width=256, p=1)
    ], p=p)


class IMG_Dataset(Dataset):

    def __init__(self,  train=True, dataset_path=None, d_type='summer', pre_process=True):

        self.im_dim = 256
        self.sens_dim = 4

        self.train_list = np.load(dataset_path + 'dataset_list.npy', allow_pickle=True)

        if train:
            self.train_list = self.train_list[:-100]
        else:
            self.train_list = self.train_list[-100:]
        self.im_list = self.train_list[:, 0]
        self.sens_list = self.train_list[:, 1]
        self.dataset_type = d_type

        self.sensordata_normalizer = pickle.load(open(dataset_path + 'scalerfile_wheel_telescope', 'rb'))
        self.dataset_path = dataset_path

        if pre_process:
            self.pre_process_data()

        self.im_list_len = len(self.im_list)
        self.sens_list_len = len(self.sens_list)

        self.transforms = self.transform_aug(im_height=self.im_dim, im_width=self.im_dim)

    def __len__(self):
        return len(self.im_list)

    def __get_sens_dim__(self):

        return self.im_dim, self.sens_dim

    def __getitem__(self, idx):
        im = np.load(self.dataset_path + self.im_list[idx])
        im = np.array(im, dtype='float32')
        im_w = im.shape[1]
        im_h = im.shape[0]

        im = self.transforms(image=im)['image']
        im = im / 255.
        img = torch.from_numpy(im.copy())
        img = img.unsqueeze_(0)

        sens_array = np.array([self.sens_list[idx]])
        if self.dataset_type == 'summer':
            # print(sens_array)
            sens = sens_array[:, [1, 2, 8, 16, 17]]
            act = sens_array[:, [34, 35, 22]]
            #print(sens, act)

        # process raw data
        nx = np.zeros((1, 4))
        nx[:, 0] = sens[:, 0]
        nx[:, 1] = sens[:, 1]
        nx[:, 2] = (sens[:, 2]) / 100.
        nx[:, 3] = (abs(sens[:, 4] - sens[:, 3])) / 100.
        nx = self.sensordata_normalizer.transform(nx)[0]
        x = torch.from_numpy(nx.copy())

        action_scale = [128., 128., 50.]
        act = act.astype(np.int8)
        act[:, 0] = convert_uint_sensor(act[:, 0])
        act[:, 1] = convert_uint_sensor(act[:, 1])
        # print(act)
        act = act / action_scale
        y = torch.from_numpy(act.copy())

        # print(img.shape)
        img, x, y = img.type(torch.FloatTensor), x.type(torch.FloatTensor), y.type(torch.FloatTensor)
        return img, x, y

    def transform_aug(self, p=1, im_height=480, im_width=640):
        dropout_w = int(im_width / 8)
        dropout_h = int(im_height / 9.)
        strong = Compose(
            OneOf([
                Compose([
                    Rotate(limit=1, p=0.5),
                    RandomCrop(height=int(im_height * 0.95), width=int(im_width * 0.9), p=0.3),
                    ElasticTransform(p=0.8),
                    HorizontalFlip(p=0.5),
                    CoarseDropout(max_holes=8, max_height=dropout_w, max_width=dropout_h, min_holes=1,
                                  min_height=5, min_width=5, fill_value=0, always_apply=False, p=0.85),
                    OneOf([
                        MotionBlur(p=0.8),
                        Blur(blur_limit=20, p=0.8),
                    ], p=0.35),

                    Resize(height=256, width=256, p=1)]),

                Resize(height=256, width=256, p=1)], p=1))

        return strong

    def pre_process_data(self):
        new_im_list = []
        new_sens_list = []
        for idx, sens in enumerate(self.sens_list):
            if abs(sens[34] > 0.) or abs(sens[35] > 0.) or abs(sens[22] > 0.):
                # print(exp_single)
                new_sens_list.append(sens)
                new_im_list.append(self.im_list[idx])
            else:
                p = random.uniform(0, 1)
                if p > 0.65:
                    new_sens_list.append(sens)
                    new_im_list.append(self.im_list[idx])

        self.im_list = np.array(new_im_list)
        self.sens_list = np.array(new_sens_list)

        return


def show(b):
    import matplotlib.pyplot as plt
    plt.imshow(b)  # [112:, :], cmap='gray', vmin=0, vmax=255)
    plt.show()


def augment_and_show(aug, image):
    image = aug(image=image)['image']
    # plt.figure(figsize=(10, 10))
    print(image)
    plt.imshow(image)
    plt.show()


def test_aug():

    img = Image.open('/media/wenyan/data_nataliya/camera21062019/depth_camera1246.svo/depth000005.png')
    # img = img.resize((640, 480), Image.ANTIALIAS)
    a = np.array(img).astype('uint16')
    a = cv2.resize(a, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
    print(a.shape)
    b = a / 65525. * 255
    show(b)
    im_w = b.shape[1]
    im_h = b.shape[0]

    aug = HorizontalFlip(p=1)

    aug = CoarseDropout(max_holes=8, max_height=50, max_width=45, min_holes=1,
                        min_height=30, min_width=30, fill_value=0, always_apply=False, p=0.85)

    for i in range(10):
        # Rotate(limit=20, p=1)  # RandomCrop(height=640, width=1000, p=1)  # Blur(blur_limit=(50, 50), p=1)  # ElasticTransform()
        aug = strong_aug(im_height=im_h, im_width=im_w)
        augment_and_show(aug, b)

    """
    HorizontalFlip(p=0.5),
    CoarseDropout(max_holes=8, max_height=50, max_width=45, min_holes=1,
                  min_height=30, min_width=30, fill_value=0, always_apply=False, p=0.85),
    OneOf([
        MotionBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    RandomCrop(height=640, width=1000, p=0.5),
    Rotate(limit=20, p=0.5),
    ElasticTransform(p=0.7),
    """

if __name__ == '__main__':

    dataset = IMG_Dataset(
        dataset_path='/media/wenyan/data_nataliya/avant/data_processing/data/im_attention_dataset/')
    dataload = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(1))
"""
    for i, data in enumerate(dataload):
        img, x, y = data
        print(x, y)
        show(img.numpy()[0])

"""
"""
    b = np.load('/media/wenyan/data_nataliya/avant/data_processing/data/im_attention_dataset/im_data/camera1246scoop1/depth000195.png.npy')
    im_w = b.shape[1]
    im_h = b.shape[0]
    aug = strong_aug(im_height=im_h, im_width=im_w)
    augment_and_show(aug, b)

"""
