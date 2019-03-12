import cv2
import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from torch.utils import data
from ptsemseg.augmentations import *
def closetRGB(rgb_pixels):
    rgb_pixels = np.array(rgb_pixels)
    colourset = np.array([[0, 0, 0], [9, 8, 213], [8, 102, 11]])


    closest = np.inf
    for i, colours in enumerate(colourset):
        d = np.sqrt(
              ((colours[0] - rgb_pixels[0]) *  (colours[0] - rgb_pixels[0]))
            + ((colours[1] - rgb_pixels[1]) *  (colours[1] - rgb_pixels[1]))
            + ((colours[2] - rgb_pixels[2]) *  (colours[2] - rgb_pixels[2]))
               )
        if d < closest:
            indx = i
            closest = d

    return indx


print(torch.__version__)
class camvidDLoader(data.Dataset):
    def __init__(self, root, split="train", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.img_size = [360, 480]
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 4
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + '/' + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name
        
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)


        #lbl = m.imread(lbl_path)
        #################################################################
   
        lbl = cv2.imread(lbl_path)
        Z = lbl.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((lbl.shape))

        unique_pixels = np.vstack({tuple(r) for r in res2.reshape(-1, 3)})
        #print(unique_pixels)

        for rgb_pixels in unique_pixels:
            #print(rgb_pixels)
            # get colour
            ind = closetRGB(rgb_pixels)
            #print(ind)
            #if ind != 0:
            res2[np.where((res2 == rgb_pixels).all(axis=2))] = [ind, ind, ind]


        #################################################################
        lbl = np.array(res2, dtype=np.int8)
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)



        return img, lbl


    def transform(self, img, lbl):

        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
          
        classes = np.unique(lbl)
        print classes
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        if not np.all(np.unique(lbl) < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")
            

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = ( mask[:,:,0] / 10.0 ) * 256 + mask[:,:,1]
        return np.array(label_mask, dtype=np.uint8)

    def decode_segmap(self, temp, plot=False):
        P_root = [255, 0, 0]
        s_root = [0, 128, 0]
        Unlabelled = [0, 0, 0]

        label_colours = np.array([P_root, s_root, Unlabelled])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        print 'RGB is on'
        return rgb

if __name__ == '__main__':
    local_path = '/home/...../Dataset/CamVid'

    augmentations = Compose([RandomRotate(10),
                             RandomHorizontallyFlip()])
    
    dst = camvidDLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        plt.savefig('myfig')
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
