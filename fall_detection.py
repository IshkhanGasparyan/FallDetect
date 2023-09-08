import torch
import numpy as np

class FallDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __call__(self, skeleton_cache):
        ds = skeleton_cache
        need = np.array([0,5,6,7,8,11,12,13,14,15,16])
        ds = np.array([ds[:, i,:] for i in need])
        ds = np.array([[ds[i][j] for i in range(ds.shape[0])] for j in range(ds.shape[1])], dtype=float)
        ds = ds[~np.all(ds<=0 , axis = (1,2))]
        vec = np.zeros((ds.shape[0], 10, 2))
        vec = np.array(
            [[(ds[i][1]+ds[i][2])/2-ds[i][0] for i in range (ds.shape[0])],
               np.array([(ds[i][5]+ds[i][6])/2-ds[i][0] for i in range (ds.shape[0])]),
               np.array([ds[i][3]-ds[i][1] for i in range (ds.shape[0])]),
               np.array([ds[i][2]-ds[i][1] for i in range (ds.shape[0])]),
               np.array([ds[i][4]-ds[i][2] for i in range (ds.shape[0])]),
               np.array([ds[i][7]-ds[i][5] for i in range (ds.shape[0])]),
               np.array([ds[i][6]-ds[i][5] for i in range (ds.shape[0])]),
               np.array([ds[i][8]-ds[i][6] for i in range (ds.shape[0])]),
               np.array([ds[i][9]-ds[i][7] for i in range (ds.shape[0])]),
               np.array([ds[i][10]-ds[i][8] for i in range (ds.shape[0])])], dtype=float)
        
        vec = np.array([[vec[i][j] for i in range(10)] for j in range(vec.shape[1])], dtype=float)

        angles = np.array([self.angle(vec[i]) for i in range(vec.shape[0])])
        ang_dif = np.squeeze(np.abs(np.array([angles[1:] - angles[:-1]])), axis=0)

        ang_mean = np.mean(ang_dif)
        
        return ang_mean > 0.1, ang_mean

    
    def angle(self, arr):
        tup = ((2,3),(3,4),(5,6),(6,7),(5,8),(7,9))
        angles = np.array([arr[0,1]/np.sqrt(arr[0]@arr[0]),arr[1,1]/np.sqrt(arr[1]@arr[1])])
        for item in tup:
                angles = np.append(angles,                        np.dot(arr[item[0]],arr[item[1]])/(np.linalg.norm(arr[item[0]])*np.linalg.norm(arr[item[1]])))
           
        return np.arccos(angles)
    
     


