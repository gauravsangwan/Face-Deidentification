import torch
import torch.nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

IMAGE_SIZE = 220
mtcnn = MTCNN(
    image_size = IMAGE_SIZE, margin = 0 ,
    thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True,
    device = 
)



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