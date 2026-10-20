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


class DummyDataset(Dataset):
    def __init__(
            self, 
            # width=1024, height=576, 
            sample_frames=25, 
            base_folder='data/samples/',
            temporal_sample=None,
            transform=None,
            seed=42,
        ):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        # self.base_folder = 'bdd100k/images/track/mini'
        self.base_folder = base_folder
        self.video_lists = glob.glob(os.path.join(self.base_folder, '*.mp4'))
        self.num_samples = len(self.video_lists)
        self.channels = 3
        # self.width = width
        # self.height = height
        self.sample_frames = sample_frames
        self.temporal_sample = temporal_sample
        self.transform = transform

        self.seed = seed
        if dist.is_initialized():
            self.rng = np.random.default_rng(seed + dist.get_rank())
        else:
            self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def get_sample(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """

        # path = random.choice(self.video_lists)
        # path = self.rng.choice(self.video_lists)
        path = self.video_lists[idx]

        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        if not end_frame_ind - start_frame_ind >= self.sample_frames:
            raise ValueError(f'video {path} does not have enough frames')
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.sample_frames, dtype=int)
        video = vframes[frame_indice]

        # (f c h w)
        pixel_values = self.transform(video)

        return {'pixel_values': pixel_values}

    def __getitem__(self, idx):
        while(True):
            try:
                # idx = np.random.randint(0, len(self.video_lists) - 1)
                idx = self.rng.integers(0, len(self.video_lists))
                item = self.get_sample(idx)
                return item
            except:
                warnings.warn(f'loading {idx} failed, retrying...')
                # idx = np.random.randint(0, len(self.video_lists) - 1)
