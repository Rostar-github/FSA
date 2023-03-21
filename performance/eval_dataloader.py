from torch.utils.data import DataLoader
from torchvision.datasets import *
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class AddGaussianNoise:
    def __init__(self, alpha, mean, std) -> None:
        self.alpha = alpha
        self.mean = mean
        self.std = std

    def __call__(self, img) -> Image:
        alpha = self.alpha * np.random.rand()
        img = np.array(img) / 255
        h, w, c = img.shape
        channel = alpha * np.random.normal(loc=self.mean, scale=self.std, size=(h, w, 1))
        noise = np.repeat(channel, c, axis=2)
        img = noise + img
        img[img > 1] = 1
        img[img < 0] = 0
        img *= 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


def get_dataloader(dataset, batch_size):
    if dataset == "mnist":
        cli_dataset = MNIST(
            "./dataset/mnist/eval",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        adv_dataset = MNIST(
            "./dataset/mnist/adv",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
    elif dataset == "fashion-mnist":
        cli_dataset = FashionMNIST(
            "./dataset/fashion_mnist/eval",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        adv_dataset = FashionMNIST(
            "./dataset/fashion_mnist/adv",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
    elif dataset == "cifar10":
        cli_dataset = CIFAR10(
            "./dataset/cifar10/eval",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        adv_dataset = CIFAR10(
            "./dataset/cifar10/adv",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
    elif dataset == "cifar10_aug":
        cli_dataset = CIFAR10(
            "./dataset/cifar10/eval",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        adv_dataset = CIFAR10(
            "./dataset/cifar10/adv",
            train=True,
            download=True,
            transform=transforms.Compose(
                [AddGaussianNoise(alpha=0.15, mean=0, std=1),
                 transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                 transforms.ToTensor()])
        )

    adv_dataloader = DataLoader(
        dataset=adv_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    cli_dataloader = DataLoader(
        dataset=cli_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    return adv_dataloader, cli_dataloader