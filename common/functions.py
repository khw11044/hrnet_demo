import numpy as np 
import torch


# def scaled_normalized2d(pose):    # 
#     scale_norm = np.sqrt(np.power(pose[:, 0:34],2).sum(axis=1, keepdim=True) / 34)                    # 3d pose도 scaling 필요 
#     p3d_scaled = pose[:, 0:34]/scale_norm
#     return p3d_scaled.reshape(-1, 2, 17).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()

# def scaled_normalized3d(pose):    # 
#     scale_norm = np.sqrt(np.power(pose[:, 0:51],2).square().sum(axis=1, keepdim=True) / 51)
#     scaled_pose = pose[:, 0:51]/scale_norm
#     return scaled_pose.reshape(-1, 3, 17).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()

def scaled_normalized2d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:34].square().sum(axis=1, keepdim=True) / 34)                    # 3d pose도 scaling 필요 
    p3d_scaled = pose[:, 0:34]/scale_norm
    return p3d_scaled.reshape(-1, 2, 17).permute(0,2,1), scale_norm.reshape(-1, 1, 1)

def scaled_normalized3d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:51].square().sum(axis=1, keepdim=True) / 51)
    scaled_pose = pose[:, 0:51]/scale_norm
    return scaled_pose.reshape(-1, 3, 17).permute(0,2,1), scale_norm.reshape(-1, 1, 1)

def regular_normalized3d(poseset):
    pose_norm_list = []

    for i in range(len(poseset)):
        root_joints = poseset[i].T[:, [0]]                                     
        pose_norm = np.linalg.norm((poseset[i].T - root_joints).reshape(-1, 51), ord=2, axis=1, keepdims=True)  
        poseset[i] = (poseset[i].T - root_joints).T                
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)

    return poseset, np.array(pose_norm_list)