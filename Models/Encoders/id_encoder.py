import torch
import torch.nn
IMAGE_SIZE = 220

class Id_encoder(nn.Module):
    def __init__(self):
        super(Id_encoder,self).__init__()
    
    #crop tensor according to the boundary boxes
    def crop_tensor(self,x):
        pass

    #preprocessing images to the id encoder
    def preprocess_imgs_to_ide(self,images):
        pass

    def forward(self,images):
        pass