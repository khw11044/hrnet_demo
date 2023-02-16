import numpy as np 

import glob 
import os 

import pickle

pickle_off = open('h36m_gt_dataset.pickle', "rb")
data = pickle.load(pickle_off)
print()

data['p2d_pred'] = {}
data['p2d_pred']['54138969'] = np.zeros([1,17,2])
data['p2d_pred']['55011271'] = np.zeros([1,17,2])
data['p2d_pred']['58860488'] = np.zeros([1,17,2])
data['p2d_pred']['60457274'] = np.zeros([1,17,2])

data['confidences'] = {}
data['confidences']['54138969'] = np.zeros([1,17])
data['confidences']['55011271'] = np.zeros([1,17])
data['confidences']['58860488'] = np.zeros([1,17])
data['confidences']['60457274'] = np.zeros([1,17])

cams = ['54138969', '55011271', '58860488', '60457274']

root = './pose2D_h36m'

subjects = glob.glob(root + '/*')

for sub in subjects:
    print(sub)
    actions = glob.glob(sub + '/*')
    
    for act in actions:
        print(act)
        views = glob.glob(act + '/*')
        
        for c, view in enumerate(views):
            print(view)
            keypoint_data = np.load(view + '/keypoints.npz')
            keypoint2D = keypoint_data['positions_2d'][:,:,:2]
            keypoint_conf = keypoint_data['positions_2d'][:,:,-1]
            
            data['p2d_pred'][cams[c]] = np.concatenate([data['p2d_pred'][cams[c]], keypoint2D], axis=0)
            data['confidences'][cams[c]] = np.concatenate([data['confidences'][cams[c]], keypoint_conf], axis=0)
            
            print()
    print()
print()