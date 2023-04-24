import cv2
import os
from enum import Enum
from albumentations import *

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2



_file_names = {
    "mask1": MaskLabels.MASK,
    "mask2": MaskLabels.MASK,
    "mask3": MaskLabels.MASK,
    "mask4": MaskLabels.MASK,
    "mask5": MaskLabels.MASK,
    "incorrect_mask": MaskLabels.INCORRECT,
    "normal": MaskLabels.NORMAL
}


'''
아래 코드에서 수행할 task
1. 60대 남성(83명)의 horizon 데이터를 추가함
2. 60대(192명)의 ShiftScaleRotate 데이터를 추가함 | num_aug 0번
3. 60대(192명)의 Sharpen 데이터를 추가함 | num_aug 1번
4. 60대(192명)의 Grid Distortion 데이터를 추가함 | num_aug 2번
(총 192 * 3 + 83 = 659개의 데이터가 추가됨)
'''

data_dir = '/opt/ml/input/data/train/images' # image 경로 입력


horizon = Compose([
            HorizontalFlip(p=1.0),
        ], p=1.0)

shift = Compose([
            ShiftScaleRotate(rotate_limit=15, p=1.0), # 15도 이상 돌아가지 않도록
        ], p=1.0)

sharpen = Compose([
            Sharpen(alpha=(0.7,1), always_apply=True)
        ])

grid = Compose([
            GridDistortion()
        ])


idx = 6960 # 기존 데이터의 제일 마지막 id가 006959

profiles = os.listdir(data_dir)
for profile in profiles: # profile : 000001_female_Asian_45
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
    id, gender, race, age = profile.split('_')

    if int(age) >= 60 and gender == 'male':
        img_folder = os.path.join(data_dir, profile) # img_folder : /opt/ml/input/data/train/images/000001_female_Asian_45
        for file_name in os.listdir(img_folder): # os.listdir(img_folder) : mask1, mask2 ...
            _file_name, ext = os.path.splitext(file_name)
            if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                continue

            img_path = os.path.join(data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
            # img_path : /opt/ml/input/data/train/images/001433_male_Asian_52/mask5.jpg

            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = horizon(image=src)['image']

            save_file_dir = f"{str(idx).zfill(6)}_{gender}_{race}_{age}"
            save_dir = os.path.join(data_dir, save_file_dir)

            save_file_path = os.path.join(data_dir, save_file_dir, file_name)
            if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            cv2.imwrite(save_file_path, img)
        idx += 1

for num_aug in range(3):
    for profile in profiles: # profile : 000001_female_Asian_45
        if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
            continue
        id, gender, race, age = profile.split('_')
        print(f"{profile} augmentation...")
        if int(age) >= 60:
            img_folder = os.path.join(data_dir, profile) # img_folder : /opt/ml/input/data/train/images/000001_female_Asian_45
            for file_name in os.listdir(img_folder): # os.listdir(img_folder) : mask1, mask2 ...
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                # img_path : /opt/ml/input/data/train/images/001433_male_Asian_52/mask5.jpg

                src = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if num_aug == 0:
                    img = shift(image=src)['image']
                elif num_aug == 1:
                    img = sharpen(image=src)['image']
                else:
                    img = grid(image=src)['image']

                save_file_dir = f"{str(idx).zfill(6)}_{gender}_{race}_{age}"
                save_dir = os.path.join(data_dir, save_file_dir)
                save_file_path = os.path.join(data_dir, save_file_dir, file_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_file_path, img)
            idx += 1
