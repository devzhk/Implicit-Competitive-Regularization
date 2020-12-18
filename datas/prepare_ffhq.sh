#! /bin/bash
python3 download_ffhq.py --images
python3 prepare_data.py --out <datapath>/ffhq256.mdb --size 256 images1024x1024/