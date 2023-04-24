import torch.nn as nn
import torch.nn.functional as F
import torchvision.models 
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

#EfficientNet (torchvision=0.8.1에는 없음, 0.13 이상!)
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.enet = torchvision.models.efficientnet_b0(pretrained = True)
        #self.classifier[-1] = nn.Linear(1280,  num_classes)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.enet(x)
        return self.fc(x)
    
# MobileNet
class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
        self.mobilenet = torchvision.models.mobilenet_v2(weights = weights)
        #self.dropout = nn.Dropout(0.25)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.relu = nn.ReLU()
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        return x

# FaceNet
class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # For a model pretrained on VGGFace2
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        self.facenet.classify = True
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8631, num_classes)                
           
    def forward(self, x):
        
        x = self.facenet(x)
        x = self.relu(x)
        return self.fc(x)

# ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights=torchvision.models.ResNet34_Weights.DEFAULT
        self.mobilenet = torchvision.models.resnet34(weights = weights)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        return x
