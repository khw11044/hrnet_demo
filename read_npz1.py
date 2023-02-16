import numpy as np 

import glob 
import os 

root = './pose2D_h36m'

subjects = glob.glob(root + '/*')

for sub in subjects:
    actions = glob.glob(sub + '/*')
    
    for act in actions:
        views = glob.glob(act + '/*')
        
        for view in views:
            img_folder = '/media/khw/T7/processed/h36m/' + '/'.join(view.split('/')[2:-1]) + '/imageSequence/' +  view.split('/')[-1]
            img_set = glob.glob(img_folder + '/*')
            keypoint_data = np.load(view + '/keypoints.npz')
            keypoint2D = keypoint_data['positions_2d']
            print(keypoint2D.shape)
            if len(keypoint2D) != len(img_set):
                print('keypoint data',keypoint2D.shape)
                print('img_set',len(img_set))