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
import os 


def choose_data(data_folder = 'flowers'):

    data_dir = data_folder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    data_transforms = {"train_transform": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                  "valid_transform": transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
                        "test_transform": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                                                               }



  
    image_datasets = { "train_set" : datasets.ImageFolder(train_dir, transform=data_transforms["train_transform"]),
                   "test_set" : datasets.ImageFolder(test_dir ,data_transforms[ "valid_transform"]),                 
                  "valid_set": datasets.ImageFolder(valid_dir, data_transforms["test_transform"])} 

    dataloaders = { "train_loader" : torch.utils.data.DataLoader(image_datasets["train_set"], batch_size=64, shuffle=True),
                   "test_loader" : torch.utils.data.DataLoader(image_datasets[ "test_set"], batch_size=32, shuffle=True),
                  "valid_loader": torch.utils.data.DataLoader(image_datasets[ "valid_set"], batch_size=32, shuffle=True)}



    return image_datasets , dataloaders



#Setup the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

archs = {"vgg11": 25088, "vgg13": 25088, "vgg16": 25088, "vgg19": 25088, "densenet121" : 1024,"densenet161":2208,"densenet201" : 1920,\
         "alexnet" : 9216, "inception_v3":2048 ,"resnet101":2048 ,"squeezenet1_0":1000 ,"resnet34":512}

#Define the network architecture
def choose_model(arch = "vgg11", input_size=1024,  output_size = 102, dropout= 0.2, hidden_units = [100,90,80], learning_rate = 0.001,epochs = 1):
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained = True)
        
    elif arch == "vgg19":
        model = models.vgg19(pretrained = True)    
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
        
        
    elif arch == "densenet201":
        model = models.densenet201(pretrained = True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained = True)
    elif arch == "inception_v3":
        model = models.inception_v3(pretrained=True)
        
    elif arch == "resnet101":
        model = models.resnet101(pretrained = True)
    elif arch == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained = True)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)    
    else:
        print("The choosen model architecture don't match {} . Plase choose a vilid one".format(arch))
        
    for param in model.parameters():
        param.requires_grad = False
        input_size = archs[arch]
        classifier = nn.Sequential(
                                  nn.Linear(input_size, hidden_units[0]),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(hidden_units[0], hidden_units[1]),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units[1], hidden_units[2]),
                                  nn.ReLU(),
                                  nn.Linear(hidden_units[2], output_size),
                                  nn.LogSoftmax(dim=1)
                                  )
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        #model.to(device)
        return model , optimizer , criterion 

image_datasets , dataloaders  = choose_data()

"""
def validation (model, dataloaders, criterion):
    model.to(device)
    accuracy = 0
    valid_loss = 0
    for inputs, labels in dataloaders["valid_loader"]:
        if device == "cuda" and source == "gpu":
            
            inputs, labels = inputs.to(device), labels.to(device)
            
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        valid_loss += batch_loss.item()
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        
        equals = top_class == labels.view(*top_class.shape)
        
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        return valid_loss, accuracy                    
"""

def train_model(model, criterion, optimizer, epochs = 1, print_every=30, data_source = dataloaders, source = "gpu"):   
    steps = 0
    #train_losses, valid_losses = [], []
    running_loss = 0
    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in data_source["train_loader"]:
            steps += 1

            if device == "cuda" and source == "gpu":

                inputs, labels = inputs.to(device), labels.to(device)
                
                #model.to(device)
            #model.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in data_source["valid_loader"]:
                        
                        if device == "cuda" and source == "gpu":
                            
                            inputs, labels = inputs.to(device) , labels.to(device)
                        
                        
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                         
                        
                    
                    #valid_loss , accuracy = validation (model, dataloaders, criterion)
                     
                #train_losses.append(running_loss/len(data_source["train_loader"]))
                #valid_losses.append(valid_loss/len(data_source["valid_loader"]))
                
                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Lost {:.3f}".format(valid_loss/len(data_source["valid_loader"])),
                       "Accuracy: {:.3f}".format(accuracy/len(data_source["valid_loader"]))
                      )
                running_loss = 0
                #model.train()
    return model              


def save_model(model, optimizer, path='checkpoint.pth', arch = 'alexnet',input_size = 9216, output_size = 102,  dropout= 0.2, epochs = 1, learning_rate = 0.001,hidden_units = [100,90,80]):
    
    model.class_to_idx = image_datasets ["train_set"].class_to_idx
    model.to(device)
    input_size = archs[arch]
    checkpoint = {
                  'arch': arch,
                  'input_size':  input_size,
                  'output_size': output_size,
                  'dropout': dropout,
                  'epochs' : epochs,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx, 
                  #'class_to_idx':image_datasets ["train_set"].class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                }

    torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model,optimizer,_ = choose_model(
                             checkpoint['arch'],
                             checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['dropout'],
                             checkpoint['hidden_units'],
                             checkpoint['learning_rate'],
                             checkpoint['epochs']
                             
                            
                            )
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    #model.to(device)
    #model.train()
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    img_tensor = torch.from_numpy(np_image)
            
    return  img_tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk = 5 , source = "gpu"):
    if device == "cuda" and source == "gpu":
        model.cuda()
    else:
        model.cpu() 

    img_tensor = process_image(image_path)
    img_tensor =  img_tensor.unsqueeze_(0)
    img_tensor =  img_tensor.float()
    
    if source == "gpu":
        with torch.no_grad():

            #output = model.forward(img_tensor.cuda())
            model.to(device)
            output = model.forward(img_tensor.to(device))
    else:

        with torch.no_grad():

            output = model.forward(img_tensor)


    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
    


# TODO: Display an image along with the top 5 classes
def check_sanity(path):
    fig, axes = plt.subplots(2,1)
    
    image = mpimg.imread(img_path)
    
    index = 1
    
    axes[0].imshow(image)
    
    axes[0].set_title(cat_to_name[str(index)])
    
    probabilities = predict(img_path, model, 5)
    
    prob = np.array(probabilities[0][0])
    
    classes = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    y = np.arange(len(classes))
    
    axes[1].barh(y, prob, align='center', color='blue')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(classes)
    axes[1].yaxis.grid(color='gray', linestyle='dashed')
    axes[1].xaxis.grid(color='gray', linestyle='dashed')
    axes[1].set(#xlabel="TOP 5 PREDICTED CLASSES",
       ylabel="PROBABILITY",
       #title="TOP 5 PREDICTED CLASSES ".lower()
    )
    axes[1].invert_yaxis()  # labels read top-to-bottom
    
    plt.savefig("top5_predicted_classess.png")


