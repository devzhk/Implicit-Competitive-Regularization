from argparse import ArgumentParser


def stylegan_parser():
    description = 'Basic parser for training styleGAN'
    parser = ArgumentParser(description=description)
    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    return parser


def cgd_trainer():
    usage = 'Parser for CGD training'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--datapath', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='DCGAN')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=96)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='ACGD')
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, default='WGAN')
    parser.add_argument('--d_penalty', type=float, default=0.0)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--show_iter', type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='test')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--atol', type=float, default=1e-16)
    parser.add_argument('--startn', type=int, default=0)
    parser.add_argument('--model_config', type=str, default=None)
    return parser


def train_seq_parser():
    usage = 'Parser for sequential training'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--datapath', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='DCGAN')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=96)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='WGAN')
    parser.add_argument('--d_penalty', type=float, default=0.0)
    parser.add_argument('--gp_weight', type=float, default=10)
    parser.add_argument('--d_iter', type=int, default=5,
                        help='-1: only update generator, 0: only update discriminator')
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--show_iter', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='test')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--startn', type=int, default=0)
    parser.add_argument('--model_config', type=str, default=None)
    return parser


def prepare_parser():
    usage = 'Parser for training'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10')
    parser.add_argument(
        '--datapath', type=str, default='cifar10'
    )
    parser.add_argument(
        '--model', type=str, default='DCGAN')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--startPoint', type=int, default=0)
    parser.add_argument(
        '--dropout', action='store_true', default=False)
    parser.add_argument(
        '--z_dim', type=int, default=128)
    parser.add_argument(
        '--batchsize', type=int, default=64)

    parser.add_argument(
        '--optimizer', type=str, default='Adam')
    parser.add_argument(
        '--lr_d', type=float, default=2e-4)
    parser.add_argument(
        '--lr_g', type=float, default=2e-4)
    parser.add_argument(
        '--momentum', type=float, default=0.9)
    parser.add_argument(
        '--loss_type', type=str, default='WGAN')
    parser.add_argument(
        '--g_penalty', type=float, default=0.0)
    parser.add_argument(
        '--d_penalty', type=float, default=0.0)
    # parser.add_argument('--use_gp', action='store_true', default=False)
    parser.add_argument(
        '--gp_weight', type=float, default=0.0)
    parser.add_argument(
        '--d_iter', type=int, default=1)

    parser.add_argument(
        '--epoch_num', type=int, default=600)
    parser.add_argument(
        '--show_iter', type=int, default=500)
    parser.add_argument(
        '--eval_iter', type=int, default=5000)

    parser.add_argument(
        '--eval_is', action='store_true', default=False)
    parser.add_argument(
        '--eval_fid', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default='DC-WGAN-GP')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--collect_info', action='store_true', default=False)
    return parser


def eval_parser():
    usage = 'Parser for eval'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--begin', type=int, default=4000)
    parser.add_argument('--end', type=int, default=400000)
    parser.add_argument('--step', type=int, default=4000)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument(
        '--eval_is', action='store_true', default=False)
    parser.add_argument(
        '--eval_fid', action='store_true', default=False)
    return parser