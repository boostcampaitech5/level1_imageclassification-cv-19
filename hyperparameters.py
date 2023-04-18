
from datetime import datetime
from pytz import timezone

now = datetime.now(timezone('Asia/Seoul'))
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S') 

seed = 42
epochs = 1
dataset = 'MaskBaseDataset'
augmentation ='BaseAugmentation'
resize = [128, 96]
batch_size=64
valid_batch_size = 64
model = 'BaseModel'
optimizer = 'Adam'
lr = 1e-3
val_ratio=0.2
criterion = 'cross_entropy'
lr_decay_step = 20
log_interval=20
name=folder_name
data_dir = '/opt/ml/input/data/train/images'
model_dir =  '/opt/ml/results'
