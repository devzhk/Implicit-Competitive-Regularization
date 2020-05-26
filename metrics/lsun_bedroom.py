import numpy as np
import torch
import tensorflow as tf
# from tflib.inception_score import get_inception_score
from .inception_tf13 import get_inception_score
import tflib.fid as fid

BATCH_SIZE = 100
N_CHANNEL = 3
RESOLUTION = 64
NUM_SAMPLES = 50000


def cal_inception_score(G, device, z_dim):
    all_samples = []
    samples = torch.randn(NUM_SAMPLES, z_dim)
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        samples_100 = samples[i:i + BATCH_SIZE]
        samples_100 = samples_100.to(device=device)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)) #.transpose(0, 2, 3, 1)
    return get_inception_score(all_samples)


def cal_inception_score_o(G, device, z_dim):
    all_samples = []
    samples = torch.randn(NUM_SAMPLES, z_dim)
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        samples_100 = samples[i:i + BATCH_SIZE]
        samples_100 = samples_100.to(device=device)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)) #.transpose(0, 2, 3, 1)
    return get_inception_score(list(all_samples))


def cal_fid_score(G, device, z_dim):
    stats_path = 'tflib/data/fid_stats_lsun_train.npz'
    inception_path = fid.check_or_download_inception('tflib/model')
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    fid.create_inception_graph(inception_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    all_samples = []
    samples = torch.randn(NUM_SAMPLES, z_dim, 1, 1)
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        samples_100 = samples[i:i + BATCH_SIZE]
        samples_100 = samples_100.to(device=device)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)).transpose(0, 2, 3, 1)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(all_samples, sess, batch_size=BATCH_SIZE)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    return fid_value
