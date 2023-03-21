from torch.utils.data import DataLoader
from torchvision.datasets import *
import torchvision.transforms as transforms
import os
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


def setup_dataloader(opt):
    if opt.cli_dataset == "cifar10":
        os.makedirs(opt.data_dir + "/cifar10/train", exist_ok=True)
        cifar10_train = CIFAR10(
            opt.data_dir + "/cifar10/train",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()]
            ),
        )

        os.makedirs(opt.data_dir + "/cifar10/eval", exist_ok=True)
        cifar10_eval = CIFAR10(
            opt.data_dir + "/cifar10/eval",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        )

        cli_train_dataloader = DataLoader(
            dataset=cifar10_train,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

        cli_eval_dataloader = DataLoader(
            dataset=cifar10_eval,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.cli_dataset == "fashion-mnist":
        os.makedirs(opt.data_dir + "/fashion_mnist/train", exist_ok=True)
        fashion_mnist_train = FashionMNIST(
            opt.data_dir + "/fashion_mnist/train",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()]))

        os.makedirs(opt.data_dir + "/fashion_mnist/eval", exist_ok=True)
        fashion_mnist_eval = FashionMNIST(
            opt.data_dir + "/fashion_mnist/eval",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.ToTensor()]
            ),
        )

        cli_train_dataloader = DataLoader(
            dataset=fashion_mnist_train,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

        cli_eval_dataloader = DataLoader(
            dataset=fashion_mnist_eval,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.cli_dataset == "mnist":
        os.makedirs(opt.data_dir + "/mnist/train", exist_ok=True)
        mnist_train = MNIST(
            opt.data_dir + "/mnist/train",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()]))

        os.makedirs(opt.data_dir + "/mnist/eval", exist_ok=True)
        mnist_eval = MNIST(
            opt.data_dir + "/mnist/eval",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.ToTensor()]
            ),
        )

        cli_train_dataloader = DataLoader(
            dataset=mnist_train,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

        cli_eval_dataloader = DataLoader(
            dataset=mnist_eval,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.adv_dataset == "cifar10":
        os.makedirs(opt.data_dir + "/cifar10/adv", exist_ok=True)
        adv_dataset = CIFAR10(
            opt.data_dir + "/cifar10/adv",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        )

        adv_train_dataloader = DataLoader(
            dataset=adv_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.adv_dataset == "cifar10_aug":
        os.makedirs(opt.data_dir + "/cifar10/adv", exist_ok=True)
        adv_dataset = CIFAR10(
            opt.data_dir + "/cifar10/adv",
            train=True,
            download=True,
            transform=transforms.Compose(
                [AddGaussianNoise(alpha=0.15, mean=0, std=1),
                 transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                 transforms.ToTensor()]
            ),
        )

        adv_train_dataloader = DataLoader(
            dataset=adv_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.adv_dataset == "mnist":
        os.makedirs(opt.data_dir + "/mnist/adv", exist_ok=True)
        adv_dataset = MNIST(
            opt.data_dir + "/mnist/adv",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.ToTensor()]
            ))

        adv_train_dataloader = DataLoader(
            dataset=adv_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    if opt.adv_dataset == "fashion-mnist":
        os.makedirs(opt.data_dir + "/fashion_mnist/adv", exist_ok=True)
        adv_dataset = FashionMNIST(
            opt.data_dir + "/fashion_mnist/adv",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size),
                 transforms.ToTensor()]
            ),
        )

        adv_train_dataloader = DataLoader(
            dataset=adv_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True
        )

    return cli_train_dataloader, cli_eval_dataloader, adv_train_dataloader