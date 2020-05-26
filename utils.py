from argparse import ArgumentParser


def train_seq_parser():
    usage = 'Parser for sequential training'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--datapath', type=str, default='./datas')
    parser.add_argument('--model', type=str, default='DCGAN')
    parser.add_argument('--model_weight', type=str, default='random')
    parser.add_argument('--z_dim', type=int, default=96)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--optimizer', type=int, default='Adam')
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