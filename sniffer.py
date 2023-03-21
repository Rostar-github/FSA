import torch
import torch.nn as nn
import os


class Sniffer:
    def __init__(self, dataloader, models, opt, device):
        self.device = device
        self.dataloader = dataloader
        self.auto_lr = opt.auto_lr
        self.d_lr = opt.d_lr
        self.g_lr = opt.g_lr
        self.alpha = opt.alpha
        self.gp_lambda = opt.gp_lambda
        self.d_x2y = models.Discriminator(level=opt.level).to(device)
        self.d_y2x = models.Discriminator(level=opt.level).to(device)

        self.gnet_x2y = models.Gnet(level=opt.level).to(device)
        self.gnet_y2x = models.Gnet(level=opt.level).to(device)

        if opt.load_models:
            self.encoder = torch.load(os.path.join(opt.save_path, f"encoder_{opt.level}.pth"))
            self.decoder = torch.load(os.path.join(opt.save_path, f"decoder_{opt.level}.pth"))
        else:
            self.encoder = models.Encoder(img_channels=opt.img_channels, level=opt.level).to(device)
            self.decoder = models.Decoder(img_channels=opt.img_channels, level=opt.level).to(device)
        self.optim_d_x2y = torch.optim.Adam(self.d_x2y.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        self.optim_d_y2x = torch.optim.Adam(self.d_y2x.parameters(), lr=self.d_lr, betas=(0.5, 0.999))

        self.optim_gnet_x2y = torch.optim.Adam(self.gnet_x2y.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.optim_gnet_y2x = torch.optim.Adam(self.gnet_y2x.parameters(), lr=self.g_lr, betas=(0.5, 0.999))

        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.auto_lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.auto_lr)

        self.d_x2y_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim_d_x2y, gamma=0.99)
        self.d_y2x_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim_d_y2x, gamma=0.99)

        self.gnet_x2y_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim_gnet_x2y, gamma=0.99)
        self.gnet_y2x_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim_gnet_y2x, gamma=0.99)
        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def get_batch_img(self):
        img, _ = next(iter(self.dataloader))
        return img.to(self.device)

    def attack(self, smashed_data, critic=True, pretrain=True):
        # train autoencoder to get target space y
        if pretrain:
            img = self.get_batch_img()[:smashed_data.shape[0], :, :, :]  # align data
            self.encoder.zero_grad(), self.decoder.zero_grad()
            y_space = self.encoder(img)
            rec_img = self.decoder(y_space)
            mse_loss = self.mse(img, rec_img)
            mse_loss.backward()
            self.optim_encoder.step(), self.optim_decoder.step()
            return {"mse_loss": mse_loss.item()}
        else:
            img = self.get_batch_img()[:smashed_data.shape[0], :, :, :]  # align data
            with torch.no_grad():
                y_space = self.encoder(img)
            x_space = smashed_data.detach()

            x_x2y = self.gnet_x2y(x_space)
            x_y2x = self.gnet_y2x(x_x2y)
            y_y2x = self.gnet_y2x(y_space)
            y_x2y = self.gnet_x2y(y_y2x)

            self.d_x2y.zero_grad(), self.d_y2x.zero_grad()
            # train D_x2y
            x2y_real_cost = torch.mean(self.d_x2y(y_space).view(-1))
            x2y_fake_cost = torch.mean(self.d_x2y(x_x2y.detach()).view(-1))
            d_x2y_gp_cost = self.gradient_penalty(d_net=self.d_x2y, real_samples=y_space, fake_samples=x_x2y.detach())
            d_x2y_loss = -(x2y_real_cost - x2y_fake_cost) + self.gp_lambda * d_x2y_gp_cost
            d_x2y_loss.backward()
            # train D_y2x
            y2x_real_cost = torch.mean(self.d_y2x(x_space).view(-1))
            y2x_fake_cost = torch.mean(self.d_y2x(y_y2x.detach()).view(-1))
            d_y2x_gp_cost = self.gradient_penalty(d_net=self.d_y2x, real_samples=x_space, fake_samples=y_y2x.detach())
            d_y2x_loss = -(y2x_real_cost - y2x_fake_cost) + self.gp_lambda * d_y2x_gp_cost
            d_y2x_loss.backward()
            self.optim_d_x2y.step(), self.optim_d_y2x.step()

            w_distance_x2y = x2y_real_cost - x2y_fake_cost
            w_distance_y2x = y2x_real_cost - y2x_fake_cost

            self.gnet_x2y.zero_grad(), self.optim_gnet_y2x.zero_grad()

            if critic:
                x_g_loss = -torch.mean(self.d_x2y(x_x2y).view(-1))
                x_g_loss.backward(retain_graph=True)

                y_g_loss = -torch.mean(self.d_y2x(y_y2x).view(-1))
                y_g_loss.backward(retain_graph=True)

            x_l1_loss = self.l1_loss(x_space, x_y2x) * self.alpha
            x_l1_loss.backward()

            y_l1_loss = self.l1_loss(y_space, y_x2y) * self.alpha
            y_l1_loss.backward()

            self.optim_gnet_x2y.step(), self.optim_gnet_y2x.step()

            return {"x cycle-loss": x_l1_loss.item(), "y cycle-loss": y_l1_loss.item(),
                    "w_distance_x2y": w_distance_x2y.item(), "w_distance_y2x": w_distance_y2x.item()}

    def gradient_penalty(self, d_net, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        # align batch size for fit opacus sampler
        if real_samples.size(0) == fake_samples.size(0):
            align_batchsize = real_samples.size(0)
        else:
            align_batchsize = min(real_samples.size(0), fake_samples.size(0))
            real_samples = real_samples[:align_batchsize]
            fake_samples = fake_samples[:align_batchsize]
        alpha = torch.rand((align_batchsize, 1, 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = d_net(interpolates)
        fake = torch.full((align_batchsize, 1), 1.0, requires_grad=False, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
