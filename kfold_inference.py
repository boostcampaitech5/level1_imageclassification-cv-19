import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model_dir_list = os.listdir(model_dir)
    oof_pred = None
    for fold_model in model_dir_list:
        fold_model_path = os.path.join(model_dir, fold_model)
        model = load_model(fold_model_path, num_classes, device).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                #pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())
            fold_pred = np.array(preds)
        n_splits = len(model_dir_list)
        # 확률 값으로 앙상블을 진행하기 때문에 'k'개로 나누어줍니다. 
        if oof_pred is None:
            oof_pred = fold_pred / n_splits
        else:
            oof_pred += fold_pred / n_splits

    

    info['ans'] = np.argmax(oof_pred, axis=1)
    save_path = os.path.join(output_dir, f'output.csv')
    # save_path = os.path.join(output_dir, f'{args.model_name}.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=int, nargs="+", default=(128, 96), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='MyModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    # parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    #parser.add_argument('--model_name', type=str, default=os.environ.get('SM_CHANNEL_MODEL', 'exp'))
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir #os.path.join('./results', args.model_name)
    output_dir = model_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    
    #python3 kfold_inference.py --model 'MobileNet' --model_dir '/opt/ml/results/2023-04-19-21:36:46'
