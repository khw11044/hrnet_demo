import numpy as np 
import cv2
import matplotlib
import matplotlib.pyplot as plt 
import glob
import os 
from tqdm import tqdm

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], 
                   [0, 4], [4, 5], [5, 6], 
                   [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], 
                   [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    ccolor = (0, 150, 0)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start_conf = kps[c[0]][-1]
        end_conf = kps[c[1]][-1] * 2
        start = list(start)
        end = list(end)
        if j in [0,1,2,13,14,15]:
            color = rcolor
        elif j in [3,4,5,10,11,12]:
            color = lcolor
        else:
            color = ccolor

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness)
        if start_conf >= 0.9:
            start_color = (255, 255, 255)
        else:
            start_color = (0, 255, 0)
        
        if end_conf >= 0.9:
            end_conf = (255, 255, 255)
        else:
            end_conf = (0, 255, 0)

        cv2.circle(img, (start[0], start[1]), thickness=-1, color=start_color, radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=end_conf, radius=3)
    # cv2.circle(img, list(kps[0][:2].astype(int)), thickness=-1, color=(125, 255, 125), radius=10)
    return img


def show3Dpose(vals, ax, RADIUS):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)
    ccolor='g'

    I = np.array( [0, 0, 1, 4, 2, 5,  0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6,  7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1,  0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    RL = np.array([0, 1, 0, 1, 0, 1,  2, 2, 0,   1,  0,  0,  1,  1, 2, 2])



    for i in np.arange( len(I) ):
        if RL[i] == 0:
            color = rcolor 
        elif RL[i] == 1:
            color = lcolor
        else: 
             color = ccolor
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=color)

    # RADIUS = 0.72
    # RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def img2video(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)




def synthesis2D_3Dimages(output_dir_2D,output_dir_3D, output_dir_pose):
    ## all
    save_dir = output_dir_pose + 'pose/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        
        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(16, 8))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')