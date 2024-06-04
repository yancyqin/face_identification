#!/usr/bin/env python
# coding: utf-8

# Face detection and recognition training process
# The following example demonstrates how to fine-tune the InceptionResnetV1 model on your own dataset. 
# This will mainly follow the standard PyTorch training pattern.

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Define runtime parameters
    # The dataset should follow the VGGFace2/ImageNet style directory layout.
    # Change `data_dir` to the location of the dataset you want to fine-tune.
    data_dir = 'raw_images/'

    batch_size = 32
    epochs = 8
    workers = 0 if os.name == 'nt' else 8

    # Check if NVIDIA GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Define MTCNN module
    # See `help(MTCNN)` for more details.
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # Perform MTCNN face detection
    # Iterate over the DataLoader object and get the cropped faces.
    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, 'cropped_faces/'))
        for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        try:
            mtcnn(x, save_path=y)
            print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        except Exception as e:
            print(f"\nAn error occurred while processing batch {i+1}: {str(e)}")


if __name__ == "__main__":
    main()
