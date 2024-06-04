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
    data_dir = 'cropped_faces/'

    batch_size = 32
    epochs = 8
    workers = 0 if os.name == 'nt' else 8

    # Check if NVIDIA GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Perform MTCNN face detection
    # Iterate over the DataLoader object and get the cropped faces.
    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((160, 160)))

    # Define Inception Resnet V1 module
    # See `help(InceptionResnetV1)` for more details.
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)

    # Define optimizer, scheduler, dataset, and data loader
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir, transform=trans)
    
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    # Define loss and evaluation functions
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    # Train the model
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitialization')
    print('-' * 10)
    resnet.eval()
    val_loss, val_metrics = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    # For recording losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        train_loss, train_metrics = training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
        train_losses.append(train_loss)

        resnet.eval()
        val_loss, val_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
        val_losses.append(val_loss)

    # Save the model
    torch.save(resnet.state_dict(), 'model/resnet_model.pth')
    print('Model saved as resnet_model.pth')

    writer.close()

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()

if __name__ == "__main__":
    main()
