import torch.nn

class LatentMapper(nn.Module):
    def __init__(self):
        super().__init__()
        slope = 0.2
        self.model = nn.Sequential(
            nn.Linear(2560,2048),
            nn.LeakyReLU(negative_slope = slope),
            nn.Linear(2048,1024),
            nn.LeakyReLU(negative_slope = slope),
            nn.Linear(1024,512),
            nn.LeakyReLU(negative_slope = slope),
            nn.Linear(512,512)
        )
        for m in self.model:
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight, a = slope)
                nn.init.constant_(m.bias,0)
    
    def forward(self,input_data):
        return self.model(input_data)