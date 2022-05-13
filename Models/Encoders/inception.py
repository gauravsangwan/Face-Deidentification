import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Inception(nn.Module):
    def __init__(self):
        # pass
        super(Inception,self).__init__()
        full_inception = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True, aux_logits=False, init_weights=False)
        removed = list(full_inception.children())[:-1]
        self.model = nn.Sequential(*removed)
        # self.meta = { 'mean': [0.485, 0.456, 0.406],
        #               'std': [0.229, 0.224, 0.225],
        #               'input_size': [299, 299,3]
        #             #   'input_size': [3,299, 299]
        #               }
        self.preprocess = transforms.Compose([transforms.Resize(299),
                                                transforms.CenterCrop(299)])

    def forward(self, data):
        # pass
        resized_data = self.preprocess(data)
        return self.model(resized_data * 255)

# import torch
# lm = Inception()
# a = lm.forward(torch.randn(1,3,299,299))
# print(lm)
# print(a)