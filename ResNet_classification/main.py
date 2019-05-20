# --------------------------------------------------------
# Demonstration on using pre-trained ResNet by Pytorch
# --------------------------------------------------------

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from PIL import Image

#initialize the pretrained model
resnet101 = models.resnet101(pretrained=True)
resnet101.eval()

print (resnet101)

#load the image
im = Image.open("img/cat.jpg")


#crop and transform the image into pytorch format
normalize = transforms.Compose(
    [transforms.RandomSizedCrop(224, scale=(1,1)), #crop to 224x224
     transforms.ToTensor(), #crom WxHxC to CxWxH, also from 0-255 to 0-1
     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #mean and var
    ])
im_tensor = normalize(im)
im_batch = im_tensor.unsqueeze(0) #[3,224,224] to [1,3,224,224], mini-batch

#display the transformed image
to_pil = transforms.Compose([transforms.ToPILImage(), transforms.Scale(256)])
to_pil(im_tensor).show()


#classify
result = resnet101(im_batch) #return 1000 classes scores

#display top possible classes
with open("resnet_id.txt") as f: idx2label = eval(f.read()) #load the classification labels
for number, idx in enumerate(result[0].sort()[1].flip(0)[:3]):  #sort the 1000 indicies(descending), print the top 3 classes
    print (number, idx2label[idx.item()])