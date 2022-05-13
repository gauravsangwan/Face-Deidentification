import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from Configs import configs
IMAGE_SIZE = 220
mtcnn = MTCNN(
    image_size = IMAGE_SIZE, margin = 0 ,
    thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True,
    device = configs.device
)
to_pil = transforms.ToPILImage(mode = 'RGB')
crop_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE)])

resnet = InceptionResnetV1(pretrained='vggface2',classify=False).eval().to(configs.device)

class Id_encoder(nn.Module):
    def __init__(self):
        super(Id_encoder,self).__init__()

    #crop tensor according to the boundary boxes
    def crop_tensor(self,images,bboxes):
        # pass
        cropped_batches = []
        for idx,image in enumerate(images):
            try: 
                cropped_image = crop_transform(image[:, int(bboxes[idx][0][1]):int(bboxes[idx][0][3]),
                                        int(bboxes[idx][0][0]):int(bboxes[idx][0][2])].unsqueeze(0))
            except:
                cropped_image = crop_transform(image.unsqueeze(0)
                # pass
            cropped_batches.append(cropped_image)
        return torch.cat(cropped_batches,dim = 0)

    #preprocessing images to the id encoder
    def preprocess_imgs_to_ide(self,images):
        # pass
        bboxes = [mtcnn.detect(to_pil(image))[0] for image in images]
        cropped_images = self.crop_tensor(images,bboxes)
        return cropped_images

    def forward(self,images):
        # pass
        cropped_images = self.preprocess_imgs_to_ide(images)
        img_embeddings = resnet(cropped_images)
        return img_embeddings

# import torch
# lm = Id_encoder()
# a = lm.forward(torch.randn(1,3,299,299))
# print(lm)
# print(a)