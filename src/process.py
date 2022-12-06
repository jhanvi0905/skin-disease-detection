import torch.utils.data as data
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split


def load_data(data_path):
    """ Load Image Dataset and equalize it"""

    TRANSFORM_IMG = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(root=data_path, transform=TRANSFORM_IMG)

    return dataset


def prepare_data_loader(data_path, batch_size):
    """Returns Data Loader for given set of images """

    dataset = load_data(data_path)
    labels = dataset.targets

    return data.DataLoader(dataset, batch_size), labels


def prepare_train_data(data_path, batch_size, split):
    """For given data and ratio of split, create random batches, weigh instances based on class length to make dataloaders"""

    dataset = load_data(data_path)
    dataset_size = dataset.__len__()
    train_count = int(dataset_size * split)
    val_count = dataset_size - train_count
    train_dataset, valid_dataset = random_split(dataset, [train_count, val_count])

    y_train = [dataset.targets[i] for i in train_dataset.indices]
    y_valid = [dataset.targets[i] for i in valid_dataset.indices]

    class_instances = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_instances
    instance_weight = np.array([weight[t] for t in y_train])
    instance_weight = torch.from_numpy(instance_weight)

    sampler = WeightedRandomSampler(instance_weight.type('torch.DoubleTensor'), len(instance_weight), replacement=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, y_train, y_valid