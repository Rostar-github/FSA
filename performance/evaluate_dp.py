import torch
import argparse
from torchvision.datasets import *
from eval_dataloader import get_dataloader
import torchvision
import os


class Evaluator:
    def __init__(self, dataset, cli_dataloader, epsilon, delta, level, device):
        self.cli_dataloader = cli_dataloader
        self.dataset = dataset
        self.level = level
        self.device = device
        self.gen_batches = 16
        self.client_net = torch.load(
            f"./checkpoint/{self.dataset}_dp_{epsilon}_{delta}/client_model_{self.level}.pth")
        self.gnet = torch.load(f"./checkpoint/{self.dataset}_dp_{epsilon}_{delta}/gnet_{self.level}.pth")
        self.decoder = torch.load(
            f"./checkpoint/{self.dataset}_dp_{epsilon}_{delta}/decoder_{self.level}.pth")
        self.client_net.eval(), self.gnet.eval(), self.decoder.eval()
        self.mse = torch.nn.MSELoss()

    def save_rec_img(self):
        for batch, (img, _) in enumerate(self.cli_dataloader):
            if batch < self.gen_batches:
                img = img.to(self.device)
                with torch.no_grad():
                    rec_img = self.decoder(self.gnet(self.client_net(img)))
                    rec_img = torch.where(rec_img > 0, rec_img, torch.zeros_like(rec_img, device=self.device))
                    rec_img_grid = torchvision.utils.make_grid(rec_img, nrow=16, normalize=False)
                    os.makedirs(f"./performance/images/{self.dataset}_dp/rec_images", exist_ok=True)
                    torchvision.utils.save_image(rec_img_grid,
                                                 f"./performance/images/{self.dataset}_dp/rec_images/batch_{batch}.jpg")
                    img_grid = torchvision.utils.make_grid(img, nrow=16, normalize=False)
                    os.makedirs(f"./performance/images/{self.dataset}_dp/cli_images", exist_ok=True)
                    torchvision.utils.save_image(img_grid,
                                                 f"./performance/images/{self.dataset}_dp/cli_images/batch_{batch}.jpg")

    def avg_mse(self):
        average_mse = 0.
        for batch, (img, _) in enumerate(self.cli_dataloader):
            img = img.to(self.device)
            with torch.no_grad():
                smashed_data = self.client_net(img)
                rec_img = self.decoder(self.gnet(smashed_data))
                average_mse += self.mse(img, rec_img).item()
        average_mse = average_mse / len(self.cli_dataloader)
        return average_mse


if __name__ == '__main__':
    torch.manual_seed(88)
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist","cifar10"])
    opt = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = ["mnist", "fashion-mnist", "cifar10"]
    delta = 1e-5
    dp_epsilons = [0.01, 0.1, 0.5, 1.0]
    # calculate avg MSE
    for dataset in datasets:
        _, cli_dataloader = get_dataloader(dataset=dataset, batch_size=opt.batch_size)
        for epsilon in dp_epsilons:
            evaluator = Evaluator(dataset, cli_dataloader, epsilon, delta, opt.level, device=device)
            avg_mse = evaluator.avg_mse()
            print(f"dataset: {dataset} | ε: {epsilon} | δ: 1e-5 | {avg_mse:.3f}")
    # save the reconstructed image, epsilon=0.1
    for i in range(3):
        _, cli_dataloader = get_dataloader(dataset=datasets[i], batch_size=8)
        evaluator = Evaluator(datasets[i], cli_dataloader, dp_epsilons[1], delta, opt.level, device=device)
        evaluator.save_rec_img()

