import os
import warnings
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torch.distributed as dist


class PairDataset(Dataset):
    def __init__(
            self, 
            # sample_frames=25, 
            base_folder='data/canyu_results/results',
            # transform=None,
            # seed=42,
            with_pair=1,
        ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.with_pair = with_pair
        # Define the path to the folder containing video frames
        # self.base_folder = 'bdd100k/images/track/mini'
        self.base_folder = base_folder
        # self.image_lists = glob.glob(os.path.join(self.base_folder, '*', '*'))

        # gather a list of input images
        # image_list = glob.glob(os.path.join(self.base_folder, "*", "*.jpeg"))
        # sort the list according to the image name, 
        # a/1.jpg, a/2.jpg, a/3.jpg, ..., a/11.jpg
        # b/1.jpg, b/2.jpg, b/3.jpg, ..., b/11.jpg
        all_folders = glob.glob(os.path.join(self.base_folder, "*"))
        image_lists = []
        for folder in all_folders:
            image_list = glob.glob(os.path.join(folder, "*.jpeg")) + glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            image_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
            print_number = 20 if len(image_list) > 20 else len(image_list)
            print(image_list[:print_number])
            image_lists.extend(image_list)

        self.image_lists = image_lists

        self.num_samples = len(self.image_lists)
        print(f"number of images: {self.num_samples}")
        self.channels = 3
        # self.width = width
        # self.height = height
        # self.sample_frames = sample_frames
        # self.transform = transform

        # self.seed = seed
        # if dist.is_initialized():
        #     self.rng = np.random.default_rng(seed + dist.get_rank())
        # else:
        #     self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def get_sample(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """

        # path = random.choice(self.image_lists)
        # path = self.rng.choice(self.image_lists)

        image_path = self.image_lists[idx]

        if self.with_pair:
            if idx == len(self.image_lists) - 1:
                idx -= 1
            image_path_end = self.image_lists[idx + 1]
        # vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        # total_frames = len(vframes)

        # # Sampling video frames
        # start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # if not end_frame_ind - start_frame_ind >= self.sample_frames:
        #     raise ValueError(f'video {path} does not have enough frames')
        # frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.sample_frames, dtype=int)
        # video = vframes[frame_indice]

        # # read image
        # image = Image.open(path).convert('RGB')
        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        # image = image.unsqueeze(0)  # TCHW

        # # (f c h w)
        # pixel_values = self.transform(image)

        # return {'pixel_values': pixel_values, 'image_path': path}
        return_dict = {'image_path': image_path}
        if self.with_pair:
            return_dict['image_path_end'] = image_path_end
        return return_dict

    def __getitem__(self, idx):
        return self.get_sample(idx)

        # while(True):
        #     try:
        #         # idx = np.random.randint(0, len(self.image_lists) - 1)
        #         idx = self.rng.integers(0, len(self.image_lists))
        #         item = self.get_sample(idx)
        #         return item
        #     except:
        #         warnings.warn(f'loading {idx} failed, retrying...')
        #         # idx = np.random.randint(0, len(self.image_lists) - 1)
