import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torch.utils.data as data
import numpy as np
from PIL import Image
import helper
from collections import OrderedDict
from torch.autograd import Variable
import time
from random import randint
import argparse
import json 

from classifier import choose_data
from classifier import choose_model
from classifier import train_model
from classifier import save_model
from classifier import load_checkpoint
from classifier import process_image
from classifier import imshow
from classifier import predict
from classifier import  check_sanity


parser = argparse.ArgumentParser()

#Use GPU if available
parser.add_argument('--gpu', type=str , dest="gpu" , default="gpu", help='Use GPU if available')

parser.add_argument('--checkpoint', type=str,  dest="checkpoint", default="checkpoint.pth" , help='Model checkpoint to use when predicting')

#parser.add_argument('checkpoint', type = str, dest="path" , default='checkpoint.pth', help='Model checkpoint to use when predicting')

parser.add_argument('--category_names', type=str , dest="category_names", default='cat_to_name.json', help='Map cat file to name')

parser.add_argument('--top_k', type=int ,  dest="top_k", default=5,   help='Top 5 predicted classes')

#parser.add_argument('input_img', type = str, default='paind-project/flowers/test/1/image_06752.jpg',  help='image to predict')

parser.add_argument('--input_img', type=str ,  dest="input_img", default= "flowers/test/10/image_07104.jpg", help='image to predict')


args ,_ = parser.parse_known_args() 

image_path = args.input_img
source = args.gpu
path = args.checkpoint
topk = args.top_k



model = load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

predict_probs = predict(image_path, model, topk , source)

predict_labels = [cat_to_name[str(k + 1)] for k in np.array(predict_probs[1][0])]

prob = np.array(predict_probs[0][0])


k = 0
while k < 5 :
    print("{} with a probability of {}".format(predict_labels[k], prob[k]))
    k += 1

print("CONGRATULATIONS PROJECT N°2 COMPLETE, CARRY ON HARD WORK")
