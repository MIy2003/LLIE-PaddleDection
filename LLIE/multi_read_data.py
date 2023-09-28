import numpy as np
import paddle
from paddle.io import Dataset
import random
from PIL import Image
from glob import glob
import paddle.vision.transforms as transforms
import os
batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(Dataset):
    def __init__(self, img_dir):
        self.low_img_dir = img_dir
        #self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im)
        img_norm=paddle.to_tensor(img_norm)
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        #low = np.asarray(low, dtype=np.float32)
        #low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('/')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        return paddle.to_tensor(low), img_name

    def __len__(self):
        return self.count
