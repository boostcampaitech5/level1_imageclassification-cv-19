import torch.nn as nn
import torch.nn.functional as F
import torchvision.models 
from facenet_pytorch import  MTCNN, InceptionResnetV1, fixed_image_standardization, training

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


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
class ENet(nn.Module):
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

        '''
        ##freeze
        for name, param in self.facenet.named_parameters():
            if name.split('.')[1] == 'fc' :
                pass
            else :
                param.requires_grad = False
        
        ##freeze 확인         
        for name, param in self.facenet.named_parameters():
            print(name, param.requires_grad)
        '''
                
                
    def forward(self, x):
        
        x = self.facenet(x)
        x = self.relu(x)
        return self.fc(x)

# model = FaceNet(3)
# print(model)

