import torch.nn as nn
import torch.nn.functional as F
import torchvision.models 

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

# Custom Model Template
class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.backbone = torchvision.models.densenet121(torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        #self.backbone = torchvision.models.densenet121(pretrained=False)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, num_classes)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.backbone(x)
        #x = self.ReLU(x)
        #x = self.Sigmoid(x)
        #x = self.softmax(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    
# #EfficientNet (torchvision=0.8.1에는 없음, 0.13 이상!)
# class ENet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         #weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
#         self.enet = torchvision.models.efficientnet_b0(pretrained = True)
#         #self.classifier[-1] = nn.Linear(1280,  num_classes)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.25)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(1280, num_classes)

#     def forward(self, x):
#         x = self.enet(x)
#         return self.fc(x)
    
# # Custom Model Template
# class MobileNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.mobilenet = torchvision.models.mobilenet_v2(pretrained=False, num_classes=500)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.25)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(500, num_classes)

#     def forward(self, x):
#         x = self.mobilenet(x)
#         x = self.relu(x)
#         x = self.dropout1(x)
#         #x = self.avgpool(x)
#         #x = x.view(-1, 128)
#         return self.fc(x)
