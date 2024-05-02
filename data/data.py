import glob
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial import distance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

sys.path.insert(0,os.getcwd())

class DatasetLoader(Dataset):

    def __init__(self, image_dir, mask_dir, mode="train"):

        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_paths = glob.glob(os.path.join(self.mask_dir,"*.png"))
        self.mask_names = os.listdir(self.mask_dir)
        self.mode = mode

    def read_image(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        return img

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index):

        image_name = self.mask_names[index]
        mask_path = os.path.join(self.mask_dir,image_name)

        image_path = os.path.join(self.image_dir,image_name)

        transform = T.Compose([T.ToTensor()])

        mask = self.read_image(mask_path)
        mask_t = transform(mask)

        if(self.mode == "train"):
            image = self.read_image(image_path)
            image_t = transform(image)
        else:
            image_t = None

        return self.mode, image_name, image_t, mask_t


def custom_collate_fn(batch):

  image_names_l = []
  image_t_l = []
  mask_t_l = []

  for i, (mode, image_name, image_t, mask_t) in enumerate(batch):
      image_names_l.append(image_name)
      if(mode == "train"):
        image_t_l.append(image_t[None])
      mask_t_l.append(mask_t[None])

  if (mode == "train"):
    image_t_l = torch.cat(image_t_l)
  mask_t_l = torch.cat(mask_t_l)

  out = (image_names_l, image_t_l, mask_t_l)

  return out


if __name__ == '__main__':

    image_dir = "/mnt/d/warwick/datasets/CRAG/scene_graph/train_data/train/images"

    crag_dataset = CragDataset(image_dir="/mnt/d/warwick/datasets/CRAG/scene_graph/train_data/train/images",
                        mask_dir="/mnt/d/warwick/datasets/CRAG/scene_graph/train_data/train/masks",
                        trimask_dir="/mnt/d/warwick/datasets/CRAG/scene_graph/train_data/train/trimasks")


    #first = glas_dataset[0]
    dataloader = DataLoader(dataset=crag_dataset,batch_size=1,shuffle=True,num_workers=0,collate_fn=crag_collate_fn)
    # data_iter = iter(dataloader)
    # batch = data_iter.next()
    # print(batch)
    for batch in dataloader:
        image_t, mask_t, trimask_t, object_indices, object_coordinates, object_bounding_boxes, object_bounding_boxes_constructed, object_masks, object_trimasks, object_embeddings, triples = batch
        print(object_embeddings)
        exit(0)
