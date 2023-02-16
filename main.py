# 이거는 in-the-wild 이미지 데이터셋을 위한 코드 

import matplotlib.pyplot as plt
from common.vis_functions import *
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
# from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from lib.hrnet.gen_kpts import gen_sequence_kpts as hrnet_pose
import os
import numpy as np
import torch
import glob
from tqdm import tqdm


sys.path.append(os.getcwd())


plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

cfg_dir = 'lib/hrnet/experiments/'
model_dir = 'lib/checkpoint/'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    args = parser.parse_args()

    return args


def get_pose2D(args, sequence_path, output_dir,vis):

    sequence = glob.glob(sequence_path + '/*')

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(args, sequence, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    scores = np.expand_dims(scores,axis=3)
    keypoints_with_scores = np.concatenate((keypoints,scores), axis=3).squeeze(0)
    print('Generating 2D pose successful!')

    os.makedirs(output_dir, exist_ok=True)
    output_npz = output_dir + '/keypoints.npz'
    # np.savez_compressed(output_npz, reconstruction=keypoints)
    np.savez_compressed(output_npz, positions_2d=keypoints_with_scores)

    if vis:
        save_vis_folder = output_dir + '/vis_2D'
        os.makedirs(save_vis_folder, exist_ok=True)
        for f, img_path in enumerate(sequence):
            img = cv2.imread(img_path)
            img_with_pose = show2Dpose(keypoints_with_scores[f], img)
            cv2.imwrite(save_vis_folder + '/' + str(('%05d'% f)) + '.png', img_with_pose)
    # return output_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--img_dataset', type=str,
                        default='../1_Obtain_images/multi-view_imgset', help='input sequence_path')
    parser.add_argument('--save_2dpose_dataset', type=str,
                        default='./pose2D_dataset', help='input sequence_path')
    parser.add_argument('--vis', type=bool,
                        default=False, help='visual')
    
    args_1 = parser.parse_args()
    
    args_2 = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args_1.gpu

    multi_view_imgset = args_1.img_dataset
    print(multi_view_imgset)
    save_2dpose_dataset = args_1.save_2dpose_dataset
    print(save_2dpose_dataset)
    os.makedirs(save_2dpose_dataset, exist_ok=True)

    Large_category = os.listdir(multi_view_imgset)
    print()
    
    vis = args_1.vis
    for category in Large_category:
        print(category)
        target_persons = os.listdir('/'.join([multi_view_imgset, category]))
        for target_person in target_persons:
            print(target_person)
            views = glob.glob('/'.join([multi_view_imgset, category, target_person]) + '/*')
            for view in views:
                print(view)
                output_dir = save_2dpose_dataset + '/' + '/'.join(view.split('/')[3:])
                get_pose2D(args_2, view, output_dir,vis=vis)
