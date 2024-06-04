import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import ImageFolder

def get_humanDetection_dataset(data_path: str='path to your dataset'):
    tr = Compose([
        Resize((128, 128)),  
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_path, transform=tr)
    return dataset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float=0.1, data_path: str='path to your dataset'):
    dataset = get_humanDetection_dataset(data_path)

    # Split dataset into train and test sets 
    num_total = len(dataset)
    num_test = num_total // 5
    num_train = num_total - num_test

    trainset, testset = random_split(dataset, [num_train, num_test], torch.Generator().manual_seed(807))

    print(f"Trainset length: {len(trainset)}")
    print(f"Testset length: {len(testset)}")

    # Split trainset into num_partitions trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    partition_remainder = len(trainset) - sum(partition_len)
    for i in range(partition_remainder):
        partition_len[i] += 1

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(807))

    # Create dataloader with test & val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # print(f"Number of trainloaders: {len(trainloaders)}")
    # for i, loader in enumerate(trainloaders):
    #     print(f"Trainloader {i} length: {len(loader.dataset)}")

    # print(f"Number of valloaders: {len(valloaders)}")
    # for i, loader in enumerate(valloaders):
    #     print(f"Valloader {i} length: {len(loader.dataset)}")

    # print(f"Testloader length: {len(testloader.dataset)}")

    return trainloaders, valloaders, testloader
