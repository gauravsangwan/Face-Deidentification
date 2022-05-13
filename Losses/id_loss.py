import torch
import torch.nn as nn
from Models.UtilModels.encoders.model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self,pretrained_model_path):
        super(IDLoss, self).__init__()
        print('ResNet ArcFace Loading...')
        self.facenet = Backbone(input_size= 112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_model_path))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
    
    def extract_features(self, x):
        x = x[:,:,35:223,32:220]
        x = self.face_pool(x)
        x_features = self.facenet(x)
        return x_features

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_features(y)
        y_hat_features = self.extract_features(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        for i in range(n_samples):
            diff_target = y_hat_features[i].dot(y_feats[i])
            loss += 1 - diff_target
        
        return loss / n_samples