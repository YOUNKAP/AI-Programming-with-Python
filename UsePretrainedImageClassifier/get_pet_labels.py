#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: YOUNKAP NINA Duplex
# DATE CREATED:        26/07/2021                          
# REVISED DATE:         08/0/2021
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
   
    filenames = listdir(image_dir)
    #Empty dictoinnairy
    results_dic = dict()
    #Empty list
    pet_names = []
    
    for i in range(len( filenames)):
        if  filenames[i][0] != '.':
            #Remove jpg extention
            pet_label1 =  filenames[i].strip('.jpg')
            #All letter lower case
            pet_label1 =  pet_label1.lower()
            #  remove _
            pet_label1 =  pet_label1.split('_')
            
            #pet name 
            pet_name = ''

            for word in pet_label1:
                if word.isalpha():
                    pet_name += word + ' '
            pet_name = pet_name.strip()
            #pet_names.append(pet_name)


            if  filenames[i] not in results_dic:
                results_dic[ filenames[i]] = [pet_name]

    return results_dic
    
    """
    filenames = listdir(image_dir)
    #filenames = listdir("pet_images/")
    
    #Fonction to retrieve label from image_names
    def label_from_image_name(image_name):
        lower_image_name = image_name.lower()
        word_list_image_name = lower_image_name.split("_")
        image_label = ""
        for word in word_list_image_name:
            if word.isalpha():
                image_label += word + " "
        return image_label
    
    #Function to retrieve label list from the listt of file names
    def label_list (list_image):
        label_list = []
        for i in range(0, len(list_image),1):
            label = label_from_image_name(list_image[i])
            label_list.append(label)    
        return label_list

    # Creates emlabelpty dictionary named results_dic
    # Creates empty dictionary named results_dic
    results_dic = dict()
    pet_labels =  label_list(filenames)
    for idx in range(0, len(filenames), 1):
        if filenames[idx] not in results_dic:
            results_dic[filenames[idx]] = [pet_labels[idx]]
    
    return results_dic
"""
