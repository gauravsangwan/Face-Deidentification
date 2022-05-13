import torch
import torch.nn as nn
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Encoder_attributes(nn.Module):
    def __init__(self,pretrained = False):
        # pass
        super(Encoder_attributes,self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=pretrained, aux_logits=False, init_weights=False)
        self.model.fc = Identity()
        self.meta = { 'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225],
                      'input_size': [299, 299,3]
                    #   'input_size': [3,299, 299]
                      }

    def forward(self, x):
        return self.model(x)

class Identity(nn.Module):
    def __init__(self):
        # pass
        super(Identity,self).__init__()

    def forward(self, x):
        return x

# import torch
# lm = Encoder_attributes()
# a = lm.forward(torch.randn(1,3,299,299))
# print(lm)
# print(a)