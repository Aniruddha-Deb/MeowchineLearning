import os
import sys
import csv

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import torch.nn as nn
import torch.nn.functional as F

class ExpressionDataset(Dataset):
    
    def __init__(self, source):
        self.source = source
        self.images = pd.DataFrame()

        self.images['name'] = os.listdir(source)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.images.iloc[idx,0]
        img_path = os.path.join(self.source, img_name)
        img = io.read_image(img_path).float()
        return [img_name, img]

class ExpressionTypeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 8)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1900, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 6))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x


def num_to_type(row):
    types = ['prefix', 'infix', 'postfix']
    row[1] = types[row[1]]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
batch_size = 100
dataset = ExpressionDataset(source=sys.argv[1])
loader = DataLoader(dataset, batch_size=batch_size)

type_net = ExpressionTypeNetwork().to(device)
type_net.load_state_dict(torch.load('type_net_dict', map_location=device))
type_net.eval()

answers = pd.DataFrame(columns=['Image_Name', 'Label'])

with torch.no_grad():
    for image_names, images in loader:
        images = images.to(device)
        output = type_net(images)

        predicted = torch.argmax(output, dim=1)

        batch_answers = pd.DataFrame({
                'Image_Name': image_names,
                'Label': predicted
            })
        answers = answers.append(batch_answers)

answers.apply(num_to_type, axis=1)

answers.to_csv('MeowchineLearning_1.csv',index=False)