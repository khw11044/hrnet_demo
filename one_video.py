import matplotlib.pyplot as plt
from common.vis_functions import *
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import shutil
import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
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

def img2video(imgset, save_video, FPS):

    img_array = []

    for filename in imgset:
        img = cv2.imread(filename)
        # img = cv2.resize(img, (resize, resize))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(
        save_video, cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def get_pose2D(args_1, args_2):

    video_path = args_1.video
    output_dir = args_1.save_dir
    save_video = args_1.save_video
    save_2dpose = args_1.save_2dpose
    FPS = 30
    save_video_name = './' + video_path.split('/')[-1]


    print('\nGenerating 2D pose...')
    frames, keypoints, scores = hrnet_pose(args_2, video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    scores = np.expand_dims(scores,axis=3)
    keypoints_with_scores = np.concatenate((keypoints,scores), axis=3).squeeze(0)
    print('Generating 2D pose successful!')

    if save_2dpose:
        os.makedirs(output_dir, exist_ok=True)
        output_npz = output_dir + 'keypoints.npz'
        np.savez_compressed(output_npz, positions_2d=keypoints_with_scores)
        print('saving 2D pose npz successful!')

    if save_video:
        save_vis_folder = output_dir + 'vis_2D'
        os.makedirs(save_vis_folder, exist_ok=True)
        for f, frame in enumerate(frames):
            img_with_pose = show2Dpose(keypoints_with_scores[f], frame)
            cv2.imwrite(save_vis_folder + '/' + str(('%05d'% f)) + '.png', img_with_pose)
        print()
        seq = glob.glob(save_vis_folder +'/*')
        img2video(seq, save_video_name, FPS)
        shutil.rmtree(save_vis_folder)
        print('saving 2D pose video successful!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../1_Obtain_images/Brandon Stone 1 view2.mp4', help='input video')   # S9_Directions-1_0, S9_Directions-1_1, sample_video
    parser.add_argument('--gpu', type=str, default='0', help='input gpu num')
    parser.add_argument('--save_dir', type=str,
                        default='./', help='set save folder')
    parser.add_argument('--save_video', type=bool,
                        default=True, help='visual')
    parser.add_argument('--save_2dpose', type=bool,
                        default=True, help='visual')
    
    args_1 = parser.parse_args()
    
    args_2 = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args_1.gpu

    
    get_pose2D(args_1, args_2)
