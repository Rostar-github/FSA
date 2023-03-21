import torch
import argparse
from train import FSA
import warnings
warnings.simplefilter("ignore")


if __name__ == '__main__':
    torch.manual_seed(89)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--task_lr", type=float, default=1e-4, help="learning rate of desired task")
    parser.add_argument("--auto_lr", type=float, default=2e-4, help="learning rate of autoencoder")
    parser.add_argument("--d_lr", type=float, default=1e-4, help="learning rate of discriminator")
    parser.add_argument("--g_lr", type=float, default=1e-4, help="learning rate of generator")
    parser.add_argument("--gp_lambda", type=int, default=50, help="WGAN-GP coefficient of regular terms")
    parser.add_argument("--alpha", type=float, default=10, help="cycle consistency coefficient of regular terms")
    parser.add_argument("--critic", type=int, default=1, help="critic of generator")
    parser.add_argument("--cli_dataset", type=str, default="cifar10", choices=["mnist", "fashion-mnist","cifar10"], help="client private dataset")
    parser.add_argument("--adv_dataset", type=str, default="cifar10", choices=["mnist", "fashion-mnist","cifar10", "cifar10_aug"], help="adversary public dataset")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="path to downloaded dataset")
    parser.add_argument("--save_path", type=str, default="./checkpoint", help="path to save model")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--img_channels", type=int, default=3, help="number of image channels ")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3], help="level of training model, only support 1 & 2 & 3 for different dataset")
    parser.add_argument("--show_interval", type=int, default=20, help="show loss per x batches")
    parser.add_argument("--start_sniff", type=int, default=50, help="start to sniff after x epoch training on desired task")
    parser.add_argument("--load_models", type=bool, default=False, help="load the pretrained models")
    # differential privacy parameters (use pytorch opacus toolkit)
    parser.add_argument("--dp_mode", type=bool, default=True, help="apply differential privacy in training client model")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum gradient norm threshold in dp ")
    parser.add_argument("--delta", type=float, default=1e-5, help="slack coefficient in dp-(ε,δ) ")
    parser.add_argument("--epsilon", type=float, default=0.1, help="privacy budget in dp-(ε,δ) ")

    parser.add_argument("--gpu", type=bool, default=True, help="is cuda available")
    opt = parser.parse_args()
    print(opt)

    if opt.gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"{device} is available...")
    # training with dp mode or not
    if opt.dp_mode:
        fsa = FSA(opt, device, dp_mode=True)
    else:
        fsa = FSA(opt, device, dp_mode=False)
    # train
    fsa(epochs=opt.epochs)
