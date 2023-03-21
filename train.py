import torch
import torch.nn as nn
import torchvision.utils
from sniff import Sniffer
from models import cifar, mnist, fashion_mnist
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
from dataloader import setup_dataloader
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine


class Models:
    def __init__(self, encoder, decoder, g_net, d_net):
        self.Encoder = encoder
        self.Decoder = decoder
        self.Gnet = g_net
        self.Discriminator = d_net


class FSA:
    def __init__(self, opt, device, dp_mode):
        self.dp_mode = dp_mode

        self.train_dataloader, self.eval_dataloader, self.adv_dataloader = setup_dataloader(opt)
        self.cli_dataset = opt.cli_dataset
        self.adv_dataset = opt.adv_dataset

        self.critic = opt.critic
        self.task_lr = opt.task_lr

        self.level = opt.level
        self.img_channels = opt.img_channels
        self.load_models = opt.load_models
        self.show_interval = opt.show_interval
        self.start_sniff = opt.start_sniff

        self.rec_mse_eval = []  # save mse curve

        self.device = device

        if self.dp_mode:
            self.setup_dp(opt)

        self.setup_save_path(opt)

        self.setup_tb_writer()

        if self.cli_dataset == "mnist":
            self.setup_models(mnist, opt)

        if self.cli_dataset == "fashion-mnist":
            self.setup_models(fashion_mnist, opt)

        if self.cli_dataset == "cifar10":
            self.setup_models(cifar, opt)

        if self.dp_mode:
            self.setup_privacy_engine()

    def setup_save_path(self, save_opt):
        if self.dp_mode:
            save_opt.save_path += f"/{self.adv_dataset}_dp_{self.dp_epsilon}_{self.dp_delta}"
        else:
            save_opt.save_path += f"/{self.adv_dataset}"
        self.save_path = save_opt.save_path
        os.makedirs(self.save_path, exist_ok=True)

    def setup_tb_writer(self):
        if self.dp_mode:
            os.makedirs(
                f"./runs/{self.cli_dataset}_{self.adv_dataset}_level_{self.level}_dp_{self.dp_epsilon}_{self.dp_delta}",
                exist_ok=True)
            self.writer = SummaryWriter(
                f"runs/{self.cli_dataset}_{self.adv_dataset}_level_{self.level}_dp_{self.dp_epsilon}_{self.dp_delta}")
        else:
            os.makedirs(f"./runs/{self.cli_dataset}_{self.adv_dataset}_level_{self.level}", exist_ok=True)
            self.writer = SummaryWriter(f"runs/{self.cli_dataset}_{self.adv_dataset}_level_{self.level}")

    def setup_models(self, nets, adv_opt):
        self.adv_models = Models(nets.Encoder, nets.Decoder, nets.Gnet, nets.Discriminator)
        self.adversary = Sniffer(self.adv_dataloader, self.adv_models, adv_opt, self.device)

        if self.load_models:
            self.clientNet = torch.load(os.path.join(self.save_path, f"client_model_{self.level}.pth"))
            self.serverNet = torch.load(os.path.join(self.save_path, f"server_model_{self.level}.pth"))
        else:
            self.clientNet = nets.Client(img_channels=self.img_channels, num_class=10, level=self.level)
            self.serverNet = nets.Server(img_channels=self.img_channels, num_class=10, level=self.level)
            # if dp_mode = True, fix the clientNet (replace BatchNormLayers)
            if self.dp_mode:
                self.clientNet = ModuleValidator.fix(self.clientNet)
                if len(ModuleValidator.validate(self.clientNet, strict=True)) == 0:
                    print("The client model passes the validation.")
                else:
                    print("The client model dose not pass the validation.")
            self.clientNet = self.clientNet.to(self.device)
            self.serverNet = self.serverNet.to(self.device)

        self.c_optim = torch.optim.Adam(self.clientNet.parameters(), lr=self.task_lr)
        self.s_optim = torch.optim.Adam(self.serverNet.parameters(), lr=self.task_lr)
        self.ce_loss = nn.CrossEntropyLoss()

    def setup_dp(self, opt):
        self.dp_max_grad_norm = opt.max_grad_norm
        self.dp_delta = opt.delta
        self.dp_epsilon = opt.epsilon
        self.dp_epochs = opt.start_sniff

    def setup_privacy_engine(self):
        self.privacy_engine = PrivacyEngine()
        self.clientNet, self.c_optim, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.clientNet,
            optimizer=self.c_optim,
            data_loader=self.train_dataloader,
            epochs=self.dp_epochs,
            target_epsilon=self.dp_epsilon,
            target_delta=self.dp_delta,
            max_grad_norm=self.dp_max_grad_norm,
        )
        print(f"noise scale: {self.c_optim.noise_multiplier}, max_grad_norm: {self.c_optim.max_grad_norm}")

    def get_adv_batch_img(self):
        img, _ = next(iter(self.adv_dataloader))
        return img.to(self.device)

    def train_step(self, epoch):

        for batch, (img, label) in enumerate(self.train_dataloader):
            img = img.to(self.device)
            label = label.to(self.device)

            # pretrain desired task and autoencoder (online)
            if epoch < self.start_sniff:
                self.clientNet.train(), self.serverNet.train()
                self.c_optim.zero_grad(), self.serverNet.zero_grad()  # use optim.zero_grad() for opacus

                smashed_data = self.clientNet(img)
                pred_label = self.serverNet(smashed_data)
                ce_loss = self.ce_loss(pred_label, label)

                ce_loss.backward()

                self.c_optim.step(), self.s_optim.step()

                adv_loss = self.adversary.attack(smashed_data.detach(), pretrain=True)

                if batch % self.show_interval == 0:
                    batch_done = len(self.train_dataloader) * epoch + batch
                    self.writer.add_scalar("CELoss", ce_loss.item(), batch_done)

                    self.writer.add_scalar("MSE loss", adv_loss['mse_loss'], batch_done)

                    print(
                        f"[Epoch {epoch}] [Batch {batch}/{len(self.train_dataloader)}] [CELoss: {ce_loss.item():>0.4f}] "
                        f"[MSE loss: {adv_loss['mse_loss']:>0.4f}]")

            # start attack (offline)
            if epoch >= self.start_sniff:
                with torch.no_grad():
                    smashed_data = self.clientNet(img)  # load smashed data produced and saved by last training epoch

                if batch % self.critic == 0:
                    adv_loss = self.adversary.attack(smashed_data, critic=True, pretrain=False)
                else:
                    adv_loss = self.adversary.attack(smashed_data, critic=False, pretrain=False)

                if batch % self.show_interval == 0:
                    batch_done = len(self.train_dataloader) * epoch + batch

                    self.writer.add_scalars("w_distance",
                                            {"w_distance_x2y": adv_loss['w_distance_x2y'],
                                             "w_distance_y2x": adv_loss['w_distance_y2x']}, batch_done)

                    self.writer.add_scalars("cycle-loss", {"x cycle-loss": adv_loss['x cycle-loss'],
                                                           "y cycle-loss": adv_loss['y cycle-loss']}, batch_done)

                    print(f"[Epoch: {epoch}] "
                          f"[Batch {batch}/{len(self.train_dataloader)}] "
                          f"[W_distance_x2y: {adv_loss['w_distance_x2y']:>0.4f}]  "
                          f"[w_distance_y2x: {adv_loss['w_distance_y2x']:>0.4f}]")

    def eval_step(self, epoch):

        self.clientNet.eval()
        self.serverNet.eval()
        self.adversary.gnet_x2y.eval()
        self.adversary.decoder.eval()
        self.adversary.encoder.eval()

        total_corrects = 0
        total_samples = 0
        total_rec_mse = 0

        for batch, (img, label) in enumerate(self.eval_dataloader):

            img = img.to(self.device)
            label = label.to(self.device)

            total_samples += label.shape[0]

            with torch.no_grad():
                smashed_data = self.clientNet(img)
                pred_label = self.serverNet(smashed_data)
                pre_class = torch.argmax(pred_label, dim=1)
                total_corrects += torch.sum((pre_class == label).type(torch.float)).item()

                rec_img = self.adversary.decoder(self.adversary.gnet_x2y(smashed_data))
                rec_mse = nn.MSELoss()(img, rec_img)
                total_rec_mse += rec_mse.item()

                if batch == len(self.eval_dataloader) - 2:
                    visual_sd_grid = torchvision.utils.make_grid(smashed_data[0, 0, :, :], normalize=True)
                    self.writer.add_image("Visual smashed data", visual_sd_grid, epoch)
                    # display adversary recovered image and ground-truth
                    rec_img = torch.where(rec_img > 0, rec_img, torch.zeros_like(rec_img, device=self.device))
                    rec_img_grid = torchvision.utils.make_grid(rec_img[:25], nrow=5, normalize=False)
                    self.writer.add_image("Adversary recovered image", rec_img_grid, epoch)

                    img_grid = torchvision.utils.make_grid(img[:25], nrow=5, normalize=True)
                    self.writer.add_image("Client image", img_grid, epoch)
                    # display adversary reconstructed image and ground-truth
                    s_img = self.get_adv_batch_img()
                    rec_s_img = self.adversary.decoder(self.adversary.encoder(s_img))
                    rec_s_img = torch.where(rec_s_img > 0, rec_s_img, torch.zeros_like(rec_s_img, device=self.device))

                    rec_s_img_grid = torchvision.utils.make_grid(rec_s_img[:25], nrow=5, normalize=False)
                    self.writer.add_image("Adversary reconstructed image", rec_s_img_grid, epoch)

                    s_img_grid = torchvision.utils.make_grid(s_img[:25], nrow=5, normalize=True)
                    self.writer.add_image("Adversary real image", s_img_grid, epoch)

        avg_mse = total_rec_mse / len(self.eval_dataloader)
        self.rec_mse_eval.append(avg_mse)
        self.writer.add_scalar("Avg MSE", avg_mse, epoch)

        acc = 100 * total_corrects / total_samples
        self.writer.add_scalar("Desired task accuracy", acc, epoch)
        print(f"[Epoch {epoch}] --------> [Accuracy: {acc:>0.4f}%]")

        if self.dp_mode:
            now_epsilon = self.privacy_engine.get_epsilon(self.dp_delta)
            print(f"spent ε = {now_epsilon:.3f}, δ = {self.dp_delta}")

    def __call__(self, epochs):
        print(f"start training, total epochs: {epochs} ...")
        for epoch in range(epochs):
            self.train_step(epoch)
            self.eval_step(epoch)
            if epoch >= (epochs - self.start_sniff) // 2 + self.start_sniff:
                self.adversary.d_x2y_schedule.step(), self.adversary.d_y2x_schedule.step()
                self.adversary.gnet_x2y_schedule.step(), self.adversary.gnet_y2x_schedule.step()

            if epoch == self.start_sniff - 1 and not self.load_models:
                # save pretrained models
                torch.save(self.clientNet, os.path.join(self.save_path, f"client_model_{self.level}.pth"))
                torch.save(self.serverNet, os.path.join(self.save_path, f"server_model_{self.level}.pth"))
                torch.save(self.adversary.encoder, os.path.join(self.save_path, f"encoder_{self.level}.pth"))
                torch.save(self.adversary.decoder, os.path.join(self.save_path, f"decoder_{self.level}.pth"))
        # save identity mapping model
        torch.save(self.adversary.gnet_x2y, os.path.join(self.save_path, f"gnet_{self.level}.pth"))
        # save training curve (resconstruction mse)
        if self.dp_mode:
            with open(f"./performance/pickles/rec_mse_{self.adv_dataset}_{self.level}_dp_{self.dp_epsilon}_{self.dp_delta}.pickle",
                      "wb") as f:
                pickle.dump(self.rec_mse_eval, f)
        else:
            with open(f"./performance/pickles/rec_mse_{self.adv_dataset}_{self.level}.pickle", "wb") as f:
                pickle.dump(self.rec_mse_eval, f)


