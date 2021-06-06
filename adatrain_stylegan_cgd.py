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
from utils.non_leaking import augment
from utils.losses import d_logistic_loss

from utils.train_utils import accumulate, mixing_noise
from utils.argparser import stylegan_parser
from optims import ACGD


def train(args, loader, generator, discriminator, optimizer, g_ema, device):
    collect_info = True
    ckpt_dir = 'checkpoints/stylegan-acgd'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    fig_dir = 'figs/stylegan-acgd'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    loader = sample_data(loader)
    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    accs = torch.tensor([1.0 for i in range(50)])
    loss_dict = {}
    if args.gpu_num > 1:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator
    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    ada_ratio = 2
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
        optimizer.step(d_loss)

        num_correct = torch.sum(real_pred > 0) + torch.sum(fake_pred < 0)
        acc = num_correct.item() / (fake_pred.shape[0] + real_pred.shape[0])

        loss_dict["loss"] = d_loss.item()
        loss_dict["real_score"] = real_pred.mean().item()
        loss_dict["fake_score"] = fake_pred.mean().item()


        # update ada_ratio
        accs[i % 50] = acc
        acc_indicator = sum(accs) / 50
        if i % 2 == 0:
            if acc_indicator > 0.85:
                ada_ratio += 1
            elif acc_indicator < 0.75:
                ada_ratio -= 1
            max_ratio = 2 ** min(4, ada_ratio)
            min_ratio = 2 ** min(0, 4 - ada_ratio)
            if args.ada_train:
                print('Adjust lrs')
                optimizer.set_lr(lr_max=max_ratio * args.lr_d, lr_min=min_ratio * args.lr_d)

        accumulate(g_ema, g_module, accum)

        d_loss_val = loss_dict["loss"]
        real_score_val = loss_dict["real_score"]
        fake_score_val = loss_dict["fake_score"]

        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {d_loss_val:.4f}; Acc: {acc:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            )
        )
        if wandb and args.wandb:
            if collect_info:
                cgd_info = optimizer.get_info()
                wandb.log(
                    {
                        'CG iter num': cgd_info['iter_num'],
                        'CG runtime': cgd_info['time'],
                        'D gradient': cgd_info['grad_y'],
                        'G gradient': cgd_info['grad_x'],
                        'D hvp': cgd_info['hvp_y'],
                        'G hvp': cgd_info['hvp_x'],
                        'D cg': cgd_info['cg_y'],
                        'G cg': cgd_info['cg_x']
                    },
                    step=i,
                )
            wandb.log(
                {
                    "Generator": d_loss_val,
                    "Discriminator": d_loss_val,
                    "Ada ratio": ada_ratio,
                    'Generator lr': max_ratio * args.lr_d,
                    'Discriminator lr': min_ratio * args.lr_d,
                    "Rt": r_t_stat,
                    "Accuracy": acc_indicator,
                    "Real Score": real_score_val,
                    "Fake Score": fake_score_val
                },
                step=i,
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
        if i % 2000 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "d_optim": optimizer.state_dict(),
                    "args": args,
                    "ada_aug_p": ada_aug_p,
                },
                f"checkpoints/stylegan-acgd/fix{str(i).zfill(6)}.pt",
            )


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = stylegan_parser()
    parser.add_argument('--optimizer', type=str, default='ACGD')
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--ratio', type=float, default=10.0)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--atol', type=float, default=1e-16)
    parser.add_argument('--ada_train', action="store_true", default=False)
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
    print('lr_g: {}, lr_d: {}'.format(args.lr_d * args.ratio, args.lr_d))
    optimizer = ACGD(max_params=generator.parameters(),
                     min_params=discriminator.parameters(),
                     lr_max=args.lr_d * args.ratio, lr_min=args.lr_d,
                     tol=args.tol, atol=args.atol,
                     device=device,
                     beta=0.99)
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
        optimizer.load_state_dict(ckpt["d_optim"])
        del ckpt
        torch.cuda.empty_cache()

    optimizer.set_lr(lr_max=args.ratio * args.lr_d, lr_min=args.lr_d)
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
        wandb.init(project="styleganv2-ada-acgd",
                   config={'lr_d': args.lr_d,
                           'Ratio': args.ratio,
                           'Image size': args.size,
                           'Batchsize': args.batch,
                           'CG tolerance': args.tol})

    train(args, loader, generator, discriminator, optimizer, g_ema, device)