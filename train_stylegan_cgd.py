'''
Adapted from https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
'''

import os
try:
    import wandb

except ImportError:
    wandb = None

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from GANs.styleganv2 import Generator, Discriminator
from datas.dataset_utils import MultiResolutionDataset, data_sampler, sample_data
from non_leaking import augment
from losses import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize

from train_utils import requires_grad, accumulate, mixing_noise
from utils import stylegan_parser
from optims import ACGD, BCGD


def train(args, loader, generator, discriminator, optimizer, g_ema, device):
    ckpt_dir = 'checkpoints/stylegan-acgd'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    fig_dir = 'figs/stylegan-acgd'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    loader = sample_data(loader)
    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    mean_path_length = 0

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    if args.gpu_num > 1:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator
    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)

        d_loss = d_logistic_loss(real_pred, fake_pred)
        # d_loss = fake_pred.mean() - real_pred.mean()
        loss_dict["loss"] = d_loss.item()
        loss_dict["real_score"] = real_pred.mean().item()
        loss_dict["fake_score"] = fake_pred.mean().item()

        # d_regularize = i % args.d_reg_every == 0
        d_regularize = False
        if d_regularize:
            real_img_cp = real_img.clone().detach()
            real_img_cp.requires_grad = True
            real_pred_cp = discriminator(real_img_cp)
            r1_loss = d_r1_loss(real_pred_cp, real_img_cp)
            d_loss += args.r1 / 2 * r1_loss * args.d_reg_every
        loss_dict["r1"] = r1_loss.item()

        # g_regularize = i % args.g_reg_every == 0
        g_regularize = False
        if g_regularize: # TODO adapt code for nn.DataParallel
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )
            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            d_loss += weighted_path_loss
            mean_path_length_avg = mean_path_length.item()

        loss_dict["path"] = path_loss.mean().item()
        loss_dict["path_length"] = path_lengths.mean().item()

        optimizer.step(d_loss)
        # update ada_aug_p
        if args.augment and args.augment_p == 0:
            ada_augment_data = torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment += ada_augment_data
            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()
                r_t_stat = pred_signs / n_pred
                if r_t_stat > args.ada_target:
                    sign = 1
                else:
                    sign = -1
                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        accumulate(g_ema, g_module, accum)

        d_loss_val = loss_dict["loss"]
        r1_val = loss_dict['r1']
        path_loss_val = loss_dict["path"]
        real_score_val = loss_dict["real_score"]
        fake_score_val = loss_dict["fake_score"]
        path_length_val = loss_dict["path_length"]

        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
                f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            )
        )
        if wandb and args.wandb:
            wandb.log(
                {
                    "Generator": d_loss_val,
                    "Discriminator": d_loss_val,
                    "Augment": ada_aug_p,
                    "Rt": r_t_stat,
                    "R1": r1_val,
                    "Path Length Regularization": path_loss_val,
                    "Mean Path Length": mean_path_length,
                    "Real Score": real_score_val,
                    "Fake Score": fake_score_val,
                    "Path Length": path_length_val,
                }
            )
        if i % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _ = g_ema([sample_z])
                utils.save_image(
                    sample,
                    f"figs/stylegan-acgd/{str(i).zfill(6)}.png",
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
        if i % 100 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "d_optim": optimizer.state_dict(),
                    "args": args,
                    "ada_aug_p": ada_aug_p,
                },
                f"checkpoints/stylegan-acgd/{str(i).zfill(6)}.pt",
            )


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = stylegan_parser()
    parser.add_argument('--optimizer', type=str, default='ACGD')
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--atol', type=float, default=1e-16)
    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0
    args.distributed =False

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    optimizer = ACGD(max_params=generator.parameters(),
                     min_params=discriminator.parameters(),
                     lr_max=args.lr_g, lr_min=args.lr_d,
                     tol=args.tol, atol=args.atol,
                     device=device,
                     beta=0.99 ** g_reg_ratio)
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        # optimizer.load_state_dict(ckpt["d_optim"])
        # TODO: check the following two lines
        del ckpt
        torch.cuda.empty_cache()

    optimizer.set_lr(lr_max=args.lr_g, lr_min=args.lr_d)
    if args.gpu_num > 1:
        generator = nn.DataParallel(generator, list(range(args.gpu_num)))
        discriminator = nn.DataParallel(discriminator, list(range(args.gpu_num)))

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if wandb is not None and args.wandb:
        wandb.init(project="styleganv2-acgd",
                   config={'lr_d': args.lr_d,
                           'lr_g': args.lr_g,
                           'Image size': args.size,
                           'Batchsize': args.batch,
                           'CG tolerance': args.tol}
                   )
    train(args, loader, generator, discriminator, optimizer, g_ema, device)