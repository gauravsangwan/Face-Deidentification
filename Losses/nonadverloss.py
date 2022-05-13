from pytorch_msssim import ms_ssim as msssim
import torch
import torch.nn as nn

l1_criterion = nn.L1Loss(reduction='mean')
l2_criterion = nn.MSELoss(reduction='mean')

def rec_loss(attr_image,generated_images,a):
    msssim_loss = 1 - msssim(attr_image,generated_images,data_range = 1,size_average = True)
    l1_loss_value = l1_criterion(attr_images,generated_image)
    return a*msssim_loss + (1-a)*l1_loss_value

def id_loss(encoded_input_image,encoded_generated_image):
    return l1_criterion(encoded_input_image,encoded_generated_image)

def landmark_loss(input_attr_lndmarks,output_attr_lndmarks):
    return l2_criterion(input_attr_lndmarks,output_attr_lndmarks)

def l2_loss(attr_image,generated_images):
    return l2_criterion(attr_image,generated_images)