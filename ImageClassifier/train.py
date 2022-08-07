import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import numpy as np
from PIL import Image
import helper
from collections import OrderedDict
from torch.autograd import Variable
import time
from random import randint
import argparse

from classifier import choose_data
from classifier import choose_model
from classifier import train_model
from classifier import save_model



"""
def get_input_args():
    #Create parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Dataset folder as --data_dir with default 'flowers'
    parser.add_argument('--data_dir', type=str, default="./flowers/", help='Path to the folder of flowers images')
    #CNN architecture as --arch with default value "alexnet"
    parser.add_argument('--arch', type=str,dest="arch", default = "alexnet/", help='CNN model architecture')
    #Use GPU if available
    parser.add_argument('--gpu', action='store_true', default="gpu", help='Use GPU if available')
    #Set learning rate default "lr = 0.001"
    parser.add_argument('--learning_rate', type=float, dest="learning_rate", default = 0.001,  help='Learning rate')
    #Set numbers of hidden units
    parser.add_argument('--hidden_units',default = 256, type=int, dest="hidden_units", help='Number of hidden units')
    #Set the number of epochs
    parser.add_argument('--epochs', type=int, dest="epochs", default = 1, help='Number of epochs')
    #Set checkpoints
    parser.add_argument('--save_dir', type=str, dest="save_dir", default="./checkpoint.pth", help='Save trained model checkpoint to file')
    #Set dropout default 0.2
    parser.add_argument('--dropout', dest = "dropout", default = 0.2, help = "dropout number to avoid overfitting" )

    return in_args 

"""

#Create parse using ArgumentParser
parser = argparse.ArgumentParser()
# Dataset folder as --data_dir with default 'flowers'
parser.add_argument('--data_dir', type=str, dest="data_dir", default="./flowers/", help='Path to the folder of flowers images')
#Set the file where we save a checkpoint
parser.add_argument('--save_dir', type=str, dest="save_dir",default="./checkpoint.pth", help='Path to the checkpoint')
#CNN architecture as --arch with default value "alexnet"
#parser.add_argument('--arch', type=str,dest="arch", default = "densenet121", help='CNN model architecture')
parser.add_argument('--arch', type=str,dest="arch", default = "alexnet", help='CNN model architecture')
#Use GPU if available
parser.add_argument('--gpu', type=str , dest="gpu" , default="gpu", help='Use GPU if available')
#Set learning rate default "lr = 0.001"
parser.add_argument('--learning_rate', type=float, dest="learning_rate", default = 0.001,  help='Learning rate')
#Set numbers of hidden units
#parser.add_argument('--hidden_unit',default = 256, type=int, dest="hidden_unit", help='Number of hidden units')
#parser.add_argument('--hidden_units',default = [100, 90, 80], type=int, dest="hidden_units", help='List of hidden units')
parser.add_argument('--hidden_units', action='append', dest="hidden_units", nargs=3,type=int,default = [],help='List of hidden units')
#Set the number of epochs
parser.add_argument('--epochs', type=int, dest="epochs", default = 1, help='Number of epochs')
#SSet dropout to prevent overfitting
parser.add_argument('--dropout', dest = "dropout", default = 0.2, help='value of dropout' )
#The input size is defined by the CNN model architecture
parser.add_argument('--input_size',default = 1024, type=int, dest="input_size", help='Number of inputs to Neural Networ')
#The output size is 102 by default
parser.add_argument('--output_size',default = 102, type=int, dest="output_size", help='Total noumbers of out put images')
args ,_ = parser.parse_known_args()

data_folder = args.data_dir
path = args.save_dir
learning_rate = args.learning_rate
arch = args.arch
dropout = args.dropout

#hidden_units = args.hidden_units
if len(args.hidden_units) == 0:
    hidden_units = [100, 90, 80]
else:
    hidden_units = args.hidden_units[0]

source = args.gpu
epochs = args.epochs
input_size = args.input_size
output_size= args.output_size


image_datasets , dataloaders = choose_data()


model, optimizer, criterion = choose_model(arch , input_size,  output_size, dropout, hidden_units, learning_rate ,epochs)


model = train_model(model, criterion, optimizer, epochs, 30, dataloaders, source)

#train_model(model, criterion, optimizer, epochs, 30, dataloaders, source)

#save_model(path, arch, input_size , output_size,  dropout, epochs, learning_rate ,hidden_units)

save_model(model, optimizer, path, arch, input_size , output_size,  dropout, epochs, learning_rate ,hidden_units)

