import torch
import time
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn

from pathlib import Path
from EEGDataset import *
from torch.utils.data import DataLoader
from Models import *


model = torch.load('max_model.pt') #Load the model
model.eval() #Set the model to evaluation mode

#gets the first test subject data
dataset = h5py.File('sub_1.hdf','r')

# Access the data and labels
data = dataset['data']
label = dataset['label']
# If you want to convert the data and label to numpy arrays for easier manipulation
data = data[0:2].reshape(30,4,1024)
label = label[0:2].reshape(30, 1024)

# Print shapes to verify dimensions
print(f"Data shape: {data.shape}")  # Should be (trials x channels x data)

first_test_tensor = torch.tensor(data, dtype=torch.float32)
first_test_tensor = first_test_tensor.unsqueeze(1)
print(f"Reshaped to 4D: {first_test_tensor.shape}")


with torch.no_grad():
    output = model(first_test_tensor)
    #print(output)

    output = F.softmax(output, dim = 1)
    #print(output)
    
    for idx, x in enumerate(output):
        print(f'Predicted: {x.argmax()}, Truth: {label[idx].mean()-1}')

        #if x[0] > 0.99:
        #    print("Low Arousal")
        #else:
        #    print("High Arousal --------------")


    #print(output.shape)
    #threshold = 0.5
    # prediction = torch.sigmoid(output).item()
    # arousal = 'High' if prediction >= threshold else 'Low'
    # print(f'Predicted arousal: {arousal}')







    #gets the first stimulus seconds from the first test
# first_test = data.copy()
# print(f"Data shape: {first_test.shape}") 

# first_test = first_test[0:1]
# print(f"Data shape: {first_test.shape}") 

#temp = first_test[0][0][0:1023]
#first_test[0][0] = np.concatenate((first_test[0][0][:1024],))
# first_test[0][1] = first_test[0][1][0:1023]
# first_test[0][2] = first_test[0][2][0:1023]
# first_test[0][3] = first_test[0][3][0:1023]
# print(f"Data shape: {first_test[0].shape}") 

# copy_of_data = data.copy()
# copy_of_data = copy_of_data[0:2]

# breakup = copy_of_data[:, :, 0:1024]

# for i in range(14):
#     temp_data = copy_of_data[:, :, (1024*(i+1)):(1024*(i+2))]
#     breakup = np.vstack((breakup, temp_data))

# second_test = first_test.copy()
# second_test = second_test[:, :, :1024]

# first_test = first_test[:, :, 1024:2048]
# print(f"Data shape: {first_test.shape}") 

#Dfirst_test = np.vstack((first_test, second_test))