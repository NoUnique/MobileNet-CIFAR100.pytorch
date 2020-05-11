import torchvision
import torchvision.transforms as transforms


def CIFAR100(split, data_dir='/data', mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
    assert split in ('train', 'valid', 'test')
    if split == 'train':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomRotation(15),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean,
                                                             std=std)])
        dataset = torchvision.datasets.CIFAR100(root=data_dir,
                                                train=True,
                                                download=True,
                                                transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean,
                                                             std=std)])
        dataset = torchvision.datasets.CIFAR100(root=data_dir,
                                                train=False,
                                                download=True,
                                                transform=transform)
    return dataset
