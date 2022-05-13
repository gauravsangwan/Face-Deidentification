import torch 
import torch.nn as nn
from mobile_facenet import MobileFaceNet
from torchvision import transforms

PATH = 'Weights/mobilefacenet.pth.tar'

class Encoder(nn.Module):
    def __init__(self,model_dir = PATH):
        # pass
        super(Encoder, self).__init__()
        self.model = MobileFaceNet([112,112],136)
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.eval()
        self.resize = transforms.Resize(112)

    def preprocess(self,imgs):
        return self.resize(imgs)

    def forward(self):
        # pass
        resized_image = self.preprocess(imgs)
        outputs, _ = self.model(resized_image)

        batch_size = resized_image.shape[0]
        landmarks = torch.reshape(outputs*112,(batch_size,68,2))

        return outputs*112, landmarks[:,17:,:]

# import torch
# lm = Encoder()
# a = lm.forward(torch.randn(1,3,112,112))
# print(lm)
# print(a)