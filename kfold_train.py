import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from pytz import timezone
from datetime import datetime
import wandb

from sklearn.model_selection import StratifiedKFold
import pandas as pd
now = datetime.now(timezone('Asia/Seoul'))
folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    train_csv = pd.read_csv('/opt/ml/v2/train_labeled6.csv') #TODO add label to csv
    # 5개의 폴드 세트로 분리하는 StratifiedKFold 세트 생성
    skfold = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=2022)
    #save_dir = increment_path(os.path.join(model_dir, args.name))
    save_dir = os.path.join(model_dir, folder_name) # ./model/현재 날짜
    for fold, (train_ids, test_ids) in enumerate(skfold.split(X=train_csv.id, y=train_csv.label)):
        save_fold_name = folder_name + '-' + str(fold)
        save_fold_dir = os.path.join(save_dir, save_fold_name)
        #train_ids와 test_ids에서는 2700개의 데이터에서의 인덱스를 넘겨줍니다.
        #따라서 아래의 계산으로 이미지마다의 인덱스를 가져옵니다. 
        train_index = [ids*7 + i for ids in train_ids for i in range(7)]
        val_index = [ids*7 + i for ids in test_ids for i in range(7)]
        
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_index)

        # -- settings
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # -- dataset
        dataset_module = getattr(import_module("dataset"),
                                args.dataset)  # default: MaskBaseDataset
        dataset = dataset_module(
            data_dir=data_dir,
        )
        num_classes = dataset.num_classes  # 18

        # -- augmentation
        transform_module = getattr(import_module(
            "dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)

        # -- data_loader
        # train_set, val_set = dataset.split_dataset() #split_dataset()을 사용하지 않습니다.

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False, #randomsubsampler를 사용할시 suffle=False가 필요
            sampler=train_subsampler,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            dataset,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            sampler=valid_subsampler,
            pin_memory=use_cuda,
            drop_last=True,
        )
        # pretrained.requires_grad_(False)
        # pretrained.fc.requires_grad_(True)
        # -- model
        model_module = getattr(import_module(
            "model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)

        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"),
                            args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_fold_dir)
        with open(os.path.join(save_fold_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        # wandb
        wandb.init(project='Resnet_Test', entity='level1-cv19', name=f'{args.model_name}-{str(fold)}', config=vars(args))
        #wandb.run.save()
        wandb.config = args
        
        #wandb.tensorboard.patch(save=False, tensorboard_x=True)
        
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss,
                                    epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc,
                                    epoch * len(train_loader) + idx)
                    wandb.log({"Train" : {"loss": train_loss, "accuracy": train_acc}})

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach(
                        ).cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_index) #val_set이 없으므로 val_index가 대신합니다.
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_fold_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_fold_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                wandb.log({"Validation": {"loss": val_loss, "accuracy": val_acc}}) #step 옵션을 주지 않으니까 잘 출력이 되는 것을 볼 수 있습니다.
                wandb.log({"results": figure})
                print()
        #k번의 반복마다 wandb.init을 사용하므로 반복이 끝날 때 wandb.finish()를 사용합니다.        
        wandb.finish() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int,
                        default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000,
                        help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='MyModel',
                        help='model type (default: MyModel)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='folder_name',
                        help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--model_name', type=str, default='Mymodel',
                        help='wandb model name (default: Mymodel)')
    #k개의 fold로 데이터셋을 나눌 수 있습니다.
    parser.add_argument('--k', type=int, default=5,
                        help='number of folds to split (default: 5)')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)