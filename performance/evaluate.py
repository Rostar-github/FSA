import torch
import torchvision
import argparse
from torchvision.datasets import *
import pickle
import matplotlib.pyplot as plt
from eval_dataloader import get_dataloader
import os
from scipy.ndimage import gaussian_filter1d

class Evaluator:
    def __init__(self, adv_dataloader, cli_dataloader, opt, device):
        self.adv_dataloader = adv_dataloader
        self.cli_dataloader = cli_dataloader
        self.level = opt.level
        self.dataset = opt.dataset
        self.gen_batches = opt.gen_batches
        self.device = device
        self.client_net = torch.load(f"./checkpoint/{self.dataset}/client_model_{self.level}.pth")
        self.gnet = torch.load(f"./checkpoint/{self.dataset}/gnet_{self.level}.pth")
        self.encoder = torch.load(f"./checkpoint/{self.dataset}/encoder_{self.level}.pth")
        self.decoder = torch.load(f"./checkpoint/{self.dataset}/decoder_{self.level}.pth")
        self.client_net.eval(), self.gnet.eval(), self.encoder.eval(), self.decoder.eval()

    def save_rec_img(self):
        for batch, (img, _) in enumerate(self.cli_dataloader):
            if batch < self.gen_batches:
                img = img.to(self.device)
                with torch.no_grad():
                    rec_img = self.decoder(self.gnet(self.client_net(img)))
                    rec_img = torch.where(rec_img > 0, rec_img, torch.zeros_like(rec_img, device=self.device))
                    rec_img_grid = torchvision.utils.make_grid(rec_img, nrow=16, normalize=False)
                    os.makedirs(f"./performance/images/{self.dataset}/rec_images", exist_ok=True)
                    torchvision.utils.save_image(rec_img_grid, f"./performance/images/{self.dataset}/rec_images/batch_{batch}.jpg")
                    img_grid = torchvision.utils.make_grid(img, nrow=16, normalize=False)
                    os.makedirs(f"./performance/images/{self.dataset}/cli_images", exist_ok=True)
                    torchvision.utils.save_image(img_grid, f"./performance/images/{self.dataset}/cli_images/batch_{batch}.jpg")

    def save_adv_img(self):
        for batch, (img, _) in enumerate(self.adv_dataloader):
            if batch < self.gen_batches:
                img = img.to(self.device)
                img_grid = torchvision.utils.make_grid(img, nrow=16, normalize=False)
                os.makedirs(f"./performance/images/{self.dataset}/adv_images", exist_ok=True)
                torchvision.utils.save_image(img_grid, f"./performance/images/{self.dataset}/adv_images/batch_{batch}.jpg")

    def plot_mse(self):
        file_1 = open(f"./performance/pickles/rec_mse_{self.dataset}_1.pickle", "rb")
        file_2 = open(f"./performance/pickles/rec_mse_{self.dataset}_2.pickle", "rb")
        file_3 = open(f"./performance/pickles/rec_mse_{self.dataset}_3.pickle", "rb")
        rec_mse_1 = pickle.load(file_1)
        rec_mse_2 = pickle.load(file_2)
        rec_mse_3 = pickle.load(file_3)
        plt.rcParams["font.family"] = "Times New Roman"
        label = ["split level-1", "split level-2", "split level-3"]
        if self.dataset == "mnist":
            title = "Average Reconstruction MSE (MNIST)"
            ylabel = "Average Mean Squared Error (MNIST)"
        elif self.dataset == "fashion-mnist":
            title = "Average Reconstruction MSE (Fashion-MNIST)"
            ylabel = "Average Mean Squared Error (Fashion-MSE)"
        elif self.dataset == "cifar10":
            title = "Average Reconstruction MSE (CIFAR10)"
            ylabel = "Average Mean Squared Error (CIFAR10)"
        elif self.dataset == "cifar10_aug":
            title = "Average Reconstruction MSE (CIFAR10/AUGMENT)"
            ylabel = "Average Mean Squared Error (CIFAR10/AUGMENT)"
        xlabel = "Number of training epochs"

        rec_mse_1_smoothed = gaussian_filter1d(rec_mse_1, sigma=3)
        rec_mse_2_smoothed = gaussian_filter1d(rec_mse_2, sigma=3)
        rec_mse_3_smoothed = gaussian_filter1d(rec_mse_3, sigma=3)
        plt.plot(rec_mse_1_smoothed)
        plt.plot(rec_mse_2_smoothed)
        plt.plot(rec_mse_3_smoothed, color="indianred")
        plt.legend(label, loc=(11/16, 3/16))
        plt.title(title, pad=20, fontsize=20)
        plt.ylabel(ylabel, fontsize=13)
        plt.xlabel(xlabel, fontsize=15)
        plt.grid()
        plt.savefig(f"./performance/plots/avg_mse_{self.dataset}.jpg")

        file_1.close(), file_2.close(), file_3.close()


if __name__ == '__main__':
    torch.manual_seed(88)
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_batches", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10_aug", choices=["mnist", "fashion-mnist","cifar10", "cifar10_aug"])

    opt = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adv_dataloader, cli_dataloader = get_dataloader(dataset=opt.dataset, batch_size=opt.batch_size)
    evaluator = Evaluator(adv_dataloader, cli_dataloader, opt, device=device)
    evaluator.save_rec_img()
    # evaluator.save_adv_img()
    evaluator.plot_mse()
