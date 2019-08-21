import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from utils import transform


class ImageDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        with open(os.path.join(data_folder,self.split+"_depth_images.json"),'r') as j:
            self.depth_images=json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        origin_image=image
        depth_image=Image.open(self.depth_images[i])
        depth_image=depth_image.convert("L")

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Apply transformations
        image, depth_image,boxes, labels, difficulties = transform(image,depth_image, boxes, labels, difficulties,self.split)

        return image, depth_image,boxes, labels, difficulties,origin_image

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        depth_images=list()
        boxes = list()
        labels = list()
        difficulties = list()
        origin_images=list()

        for b in batch:
            images.append(b[0])
            depth_images.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])
            origin_images.append(b[5])


        images = torch.stack(images, dim=0)
        depth_images=torch.stack(depth_images,dim=0)

        return images,depth_images, boxes, labels, difficulties,origin_images # tensor (N, 3, 300, 300), 3 lists of N tensors each
