from torchvision import transforms
from datasets import video_transforms
from .ucf101_datasets import UCF101
from .dummy_datasets import DummyDataset
from .encdec_image_datasets import EncDecImageDataset
from .pair_datasets import PairDataset


def get_dataset(args):
    if args.dataset not in ["encdec_images", "pair_dataset"]:
        temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])
        dataset = UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
        return dataset

    elif args.dataset == 'dummy':
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),  # NOTE
            video_transforms.UCFCenterCropVideo(size=size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = DummyDataset(
            # width=args.width,
            # height=args.height,
            sample_frames=args.num_frames,
            base_folder=args.base_folder,
            temporal_sample=temporal_sample,
            transform=transform,
            seed=args.seed
        )
        return dataset

    elif args.dataset == "encdec_images":
        size = (args.height, args.width)
        transform = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            # video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(size=size),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        ])

        dataset = EncDecImageDataset(
            # width=args.width,
            # height=args.height,
            # sample_frames=args.num_frames,
            base_folder=args.base_folder,
            # temporal_sample=temporal_sample,
            transform=transform,
            seed=args.seed,
        )
        return dataset

    elif args.dataset == "pair_dataset":
        # size = (args.height, args.width)
        # transform = transforms.Compose([
        #     video_transforms.ToTensorVideo(), # TCHW
        #     # video_transforms.RandomHorizontalFlipVideo(),
        #     video_transforms.UCFCenterCropVideo(size=size),
        #     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
        # ])

        dataset = PairDataset(
            # width=args.width,
            # height=args.height,
            # sample_frames=args.num_frames,
            base_folder=args.base_folder,
            # temporal_sample=temporal_sample,
            # transform=transform,
            # seed=args.seed,
            with_pair=args.with_pair,
        )
        return dataset

    else:
        raise NotImplementedError(args.dataset)
