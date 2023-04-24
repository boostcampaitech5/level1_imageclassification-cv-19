# level1_imageclassification-cv-19
level1_imageclassification-cv-19 created by GitHub Classroom

## Project Overview



- Project Period 2023/04/10 ~ 2023/04/20
- Project Wrap-Up Report **[Mask Detection Wrap-up Report](https://docs.google.com/document/d/1STQpw1djeaGu-y6Z17kxUxizX55Ct22jSXZJEUQ-OuM/edit?usp=sharing)**

## 🙌 Members

| 강동화 | 박준서 | 서지희 | 장철호 | 한나영 |
| --- | --- | --- | --- | --- |
|  |  |  |  |<img src = "https://user-images.githubusercontent.com/19367749/233942047-3ed564cf-d770-4c70-8358-09a4731a227f.jpg" width="120" height="120" />|
| [@oktaylor](https://github.com/oktaylor) | [@Pjunn](https://github.com/Pjunn) | [@muyaaho](https://github.com/muyaaho) | [@JCH1410](https://github.com/JCH1410) | [@Bandi120424](https://github.com/Bandi120424) |

![profile.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/33cb71b0-e4b7-4544-a9dc-97693f7f6c22/profile.jpg)

## 🌏 Contributions

| 이름 | 학습 모델 | 추가 작업 |
| :---: | --- | --- |
| 강동화 | VGG, ViT, MobileNet | 날짜명으로 convention 통일, Data Augmentation을 통해 추가 데이터 구성, Error Analysis, 모델 앙상블 |
| 박준서 | MobileNetV2, ResNet, ConvNext Tiny | Wandb 실험 환경 구성, 교차 검증 셋팅, Error Analysis, 모델 앙상블 |
| 서지희 | EfficientNet | Notion template 구축, Error Analysis, 모델 앙상블 |
| 장철호 | DenseNet | Error Analysis, 모델 앙상블 |
| 한나영 | MobileNetV2, FaceNet | EDA, 세부 평가 지표 구성, Notion template 구축, Error Analysis, 모델 앙상블 |

## ❓ Problem Overview

COVID-19은 강력한 전염력을 가지고 있습니다. 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 코와 입을 완전히 가릴 수 있도록 마스크를 올바르게 착용해야합니다. 하지만 이러한 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다. 또한 성별과 나이를 추가적으로 분류하여 다양한 정보를 획득합니다. 이를 통해 구체적이고 명확한 감염자의 통계치를 산출하는 데 도움이 될 것입니다.

## 📂 Datasets

- Number of Classes: 18
- Number of Datasets: 4500 * 7[마스크 착용 5장, 이상하게 착용 1장, 미착용 1장]
- Labels:
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bc4c915a-95f8-40df-aa51-4dece5ef9056/Untitled.png)
    
- Image Size: (384, 512)
- Train: 2,700 * 7 (전체 데이터의 60%)
- Test: 1,800 * 7


## 📂 Annotation Files

- images
    - 번호_성별_인종_나이 이름의 폴더에 한 사람당 이미지 7장 ex) 000001_female_Asian_45
    - 마스크를 이상하게 쓴 이미지 1장 ex) incorrect_mask.jpg
    - 마스크를 안쓴 이미지 1장 ex) normal.jpg
    - 마스크를 제대로 쓴 이미지 5장 ex) mask1.jpg ~ mask5.jpg
    - height : 512
    - width : 384
- annotation
    - ImageID : 이미지의 고유 ID
    - ans : 이미지가 해당하는 class의 ID
- 

## 🖥️ ****Development Environment****

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5
