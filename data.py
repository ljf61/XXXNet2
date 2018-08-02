import numpy as np
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

def preprocess1(image1,image3):
    x1 = cv2.imread(image1)
    #x1 = x1[0:255,50:550]
    x1 = np.array(x1)
    x1 = x1.reshape((-1,225,600))
    x3 = cv2.imread(image3)
    #x3 = x3[0:255,50:550]
    x3 = np.array(x3)
    x3 = x3.reshape((-1,225,600))
    #X = np.concatenate((x1,x3),axis=0)
    return torch.FloatTensor(x1), torch.FloatTensor(x3)

def preprocess2(image):
    X1 = cv2.imread(image)
    X1 = cv2.resize(X1, (224, 600))
    X1 = cv2.cvtColor(X1, cv2.COLOR_BGR2GRAY)
    X = np.array(X1)
    X = X.reshape((1,224, 600))
    return torch.FloatTensor(X)

class MyDataset(data.Dataset):
    def __init__(self, poses, images, depths, loaderIMG=preprocess1, loaderDepth=preprocess2):
        self.images = images
        self.depths = depths
        self.poses = poses
        self.loaderIMG = loaderIMG
        self.loaderDepth = loaderDepth

    def __getitem__(self, index):
        #i = int(np.random.choice(['4','2']))
        pose1 = self.poses[index]
        pose2 = self.poses[index+2]
        image1 = self.images[index]
        image2 = self.images[index+2]
        depth = self.depths[index+1]
        img_rgbl, img_rgbr = self.loaderIMG(image1,image2)
        img_depth = self.loaderDepth(depth)
        pose1 = np.array(pose1)
        pose2 = np.array(pose2)   
        pose = pose2 - pose1
        #pose = pose2 - pose1
        #if pose[2] > 350:
        #    pose[2] = pose[2] - 360
        #if pose[2] < -350:
        #    pose[2] = pose[2] + 360
        #pose_xy = torch.FloatTensor(pose2 - pose1)
        #print(pose)
        pose = torch.FloatTensor(pose) 
        return pose, img_rgbl, img_rgbr, img_depth 

    def __len__(self):
        return len(self.depths)-2