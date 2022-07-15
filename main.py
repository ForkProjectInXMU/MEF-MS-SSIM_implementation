import time
import os
import cv2
import math
import argparse
import pprint
import torch
import numpy as np
from MEF_SSIMc import MEF_SSIMc
from MS_SSIMc import MS_SSIMc
from MS_SSIMc_Color import MS_SSIMc_Color
from config import cfg, cfg_from_list

image_path = './images'
sample_name = 'seq'
sample_image_name = 'Tower_Mertens07.png'
sample_type = 'png'

oe_path = '../oe.png'
ue_path = '../ue.png'
gt_path = '../gt.png'


def parse_args():
    parser = argparse.ArgumentParser(description='mef_ssimc')
    # args set
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    arg = parser.parse_args()
    return arg


def load_images():
    oe = cv2.imread(oe_path).astype(np.float64)
    ue = cv2.imread(ue_path).astype(np.float64)
    img_seq = torch.Tensor(np.stack([oe, ue], 3))
    gt = cv2.imread(gt_path).astype(np.float64)
    gt = torch.Tensor(gt)
    print(gt.shape)
    print(img_seq.shape)
    return img_seq, gt


if __name__ == "__main__":
    args = parse_args()
    print("Called with args:")
    print(args)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # something about the writing ...
    print("Using config:")
    pprint.pprint(cfg)

    # read a list json to do the experiment after ...

    img_seq, init_image = load_images()

    time1 = time.time()

    # mef_ssimc = MEF_SSIMc()
    # output_image, score = mef_ssimc(img_seq, init_image)

    # ms_ssimc = MS_SSIMc()
    # output_image, score = ms_ssimc(img_seq, init_image)

    ms_ssimc = MS_SSIMc_Color()
    output_image, score = ms_ssimc(img_seq, init_image)

    cv2.imwrite('./gt_new.png', output_image)
    print(score)

    time2 = time.time()
    print('time used: ' + str(time2 - time1))
