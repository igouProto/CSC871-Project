import matplotlib.pyplot as plt
import pandas as p
import torch
import numpy as np
import torch.nn as nn
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


def load_mnist_data(data_dir):
    train_data_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_data_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')


    # Load training images
    with open(train_data_path, 'rb') as f:
        magic, num_images, rows, cols = np.fromfile(f, dtype=np.dtype('>i4'), count=4)
        train_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

    # Load training labels
    with open(train_labels_path, 'rb') as f:
        magic, num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=2)
        train_labels = np.fromfile(f, dtype=np.uint8)

    # Load test images
    with open(test_data_path, 'rb') as f:
        magic, num_images, rows, cols = np.fromfile(f, dtype=np.dtype('>i4'), count=4)
        test_data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

    # Load test labels
    with open(test_labels_path, 'rb') as f:
        magic, num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=2)
        test_labels = np.fromfile(f, dtype=np.uint8)

    # Convert images to torch tensors and apply augmentation transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Normalizing the test images
    transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
   
    train_data = torch.stack([transform_train(img) for img in train_data])
    test_data = torch.stack([transform_test(img) for img in test_data])

    return train_data, train_labels, test_data, test_labels


def main():
    #datasets are already downloaded and unziped in ./dataset 
    dataset_path = './dataset'
    train_images, train_labels, test_images, test_labels = load_mnist_data(dataset_path)

    #print shapes of train_images, train_labels, test_images, test_labels
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    # Create DataLoader objects for images and labels
    train_images_dataset = TensorDataset(train_images, torch.tensor(train_labels))
    test_images_dataset = TensorDataset(test_images, torch.tensor(test_labels))

    train_loader = DataLoader(train_images_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_images_dataset, batch_size=32, shuffle=False)


if __name__ == '__main__':
    main()