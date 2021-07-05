import os
import sys
import csv

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import torch.nn as nn
import torch.nn.functional as F

class ExpressionDataset(Dataset):
    
    def __init__(self, source):
        self.source = source
        self.images = pd.DataFrame()
        self.transform = transforms.ToTensor()
        self.images['name'] = os.listdir(source)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform = transforms.ToTensor()
        img_height = 128
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.images.iloc[idx,0]
        img_path = os.path.join(self.source, img_name)
        img = Image.open(img_path)
        leftimg = self.transform(img.crop((0,0,img_height,img_height))).float()
        middleimg = self.transform(img.crop((img_height,0,img_height*2,img_height))).float()
        rightimg = self.transform(img.crop((img_height*2,0,img_height*3,img_height))).float()
        return [img_name, leftimg, middleimg, rightimg]

class DigitNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2420, 100)
        self.fc2 = nn.Linear(100, 14) 

    def forward(self, x):        
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
batch_size = 100
dataset = ExpressionDataset(source=sys.argv[1])
loader = DataLoader(dataset, batch_size=batch_size)

type_net = DigitNetwork().to(device)
type_net.load_state_dict(torch.load('value_net_dict', map_location=device))
type_net.eval()

op_str = '0123456789+-*/'
answers = pd.DataFrame(columns=['Image_Name', 'Value'])

with torch.no_grad():
    for img_names, leftimg, midimg, rightimg in loader:
        sections = [leftimg, midimg, rightimg]
        predictions = []
        values = []

        for images in sections:
            images = images.to(device)
            output = type_net(images)

            predicted = torch.argmax(output, dim=1)
            predictions.append(predicted)

        for i in range(len(predictions[0])):
            l, m, r = predictions[0][i], predictions[1][i], predictions[2][i]
            if l >= 10: # prefix
                values.append(int(eval(f'{m}{op_str[l]}{r}')))
            elif m >= 10: # infix
                values.append(int(eval(f'{l}{op_str[m]}{r}')))
            elif r >= 10: # postfix
                values.append(int(eval(f'{l}{op_str[r]}{m}')))
            else:
                values.append(0)

        batch_answers = pd.DataFrame({
                'Image_Name': img_names,
                'Value': values
            })
        answers = answers.append(batch_answers)

answers.to_csv('MeowchineLearning_2.csv',index=False)
