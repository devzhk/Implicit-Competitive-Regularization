# Adapted from https://github.com/fyu/lsun/blob/master/download.py

from __future__ import print_function, division
import argparse
from os.path import join

import subprocess
from urllib.request import Request, urlopen


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')


def download_scene(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def download_object(out_dir, category):
    url = f'http://dl.yf.io/lsun/objects/{category}.zip'
    out_path = join(out_dir, '{category}.zip')
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, 'set')
    subprocess.call(cmd)


def download_chksum(out_dir, category):
    url = f'http://dl.yf.io/lsun/objects/{category}.zip.md5'
    out_path = join(out_dir, '{category}.zip.md5')
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, 'Checksum')
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='')
    parser.add_argument('-c', '--category', default=None)
    parser.add_argument('--objects', action="store_true")
    args = parser.parse_args()
    if not args.objects:
        # download scenes data
        categories = list_categories()
        if args.category is None:
            print('Downloading', len(categories), 'categories')
            for category in categories:
                download_scene(args.out_dir, category, 'train')
                download_scene(args.out_dir, category, 'val')
            download_scene(args.out_dir, '', 'test')
        else:
            if args.category == 'test':
                download_scene(args.out_dir, '', 'test')
            elif args.category not in categories:
                print('Error:', args.category, "doesn't exist in", 'LSUN release')
            else:
                download_scene(args.out_dir, args.category, 'train')
                download_scene(args.out_dir, args.category, 'val')
    else:
        # download_object(args.out_dir, args.category)
        download_chksum(args.out_dir, args.category)


if __name__ == '__main__':
    main()
