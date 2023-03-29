import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, InterpolationMode, Normalize, RandomHorizontalFlip, Resize, ToTensor

from data.LMDB import LMDB

PrivateDatasetNames = ["FFHQ", "CELEBA-HQ", "CIFAR10"]


def build_loader(config):
    dsets = dict()
    dset_loaders = dict()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # get the train dataset
    dataset_name = config.data.source_dataset_name

    dsets['train'] = build_dataset(config, dataset_name)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} successfully build {dataset_name} dataset")

    sampler_train = DistributedSampler(dsets['train'],
                                       num_replicas=num_tasks,
                                       rank=global_rank,
                                       shuffle=True)

    dset_loaders['train'] = DataLoader(
        dataset=dsets['train'],
        sampler=sampler_train,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        shuffle=False,
    )

    return dsets, dset_loaders


def build_dataset(config, dataset_name):
    transform = build_transform(config)
    if dataset_name in PrivateDatasetNames:
        dataset = LMDB(root=config.data.data_root_path, transforms=transform)
    else:
        raise Exception(f"It not supports the dataset {dataset_name}!")
    return dataset


def build_transform(config):
    """ transform image into tensor """
    transform = Compose([
        Resize(size=[config.data.img_size, config.data.img_size], interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=config.aug.hflip),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to -1, 1
    ])
    return transform
