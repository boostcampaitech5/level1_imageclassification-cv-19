import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_f1_score
#import pre-trained model 

from dataset import MaskBaseDataset
from loss import create_criterion

from pytz import timezone
from datetime import datetime

##Experiment Toolkit
import wandb

#hyperparemeters
import hyperparameters as hp

now = datetime.now(timezone('Asia/Seoul'))
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S') 


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

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
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

def acc_per_class(pred, labels, num_classes): #pred, labels는 torch 

    labels_per_class = {i: 0 for i in range(num_classes)} #label안에 각 class의 갯수 (정답 갯수)
    labels_hit = {i: 0 for i in range(num_classes)}
    label_arr = labels.clone().detach().cpu().numpy()
    pred_arr = pred.clone().detach().cpu().numpy()
    
    for i in range(pred.shape[0]):    
        labels_per_class[label_arr[i]] += 1
        if pred_arr[i] == label_arr[i]:
            labels_hit[label_arr[i]] += 1
    
    class_acc = {i:0 for i in range(num_classes)} #class 별 acc 값     
    for i in range(num_classes):
        if labels_per_class[i] == 0:
            class_acc[i] = 0
        else:
            class_acc[i] = np.round(float(labels_hit[i])/float(labels_per_class[i]), 3)
    
    return class_acc

def train(data_dir, model_dir, args):
    seed_everything(args.seed)    
    wandb.init(project=args.model, entity='level1-cv19', name=args.name, config=vars(args))
    wandb.tensorboard.patch(save=False, tensorboard_x=True)
    wandb.config = args
    
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # MaskBaseDataset = 18, AgeGenderDataset = 6
    
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)

    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion, classes=num_classes)  # default: cross_entropy        
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-4)
    
    # -- logging
    
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # for calculating label acc per class 
    def concat_label(labels, mask_label, gen_label, age_label):
        mask_lb, gen_lb, age_lb = dataset.decode_multi_class(labels) 
        mask_label = torch.concat((mask_label, mask_lb.to(mask_label.device)), dim = 0)
        gen_label = torch.concat((gen_label, gen_lb.to(gen_label.device)), dim = 0)
        age_label = torch.concat((age_label, age_lb.to(age_label.device)), dim = 0)
        
        return mask_label, gen_label, age_label

    def concat_single_label(labels, single_label):
        single_lb = dataset.decode_multi_class(labels) 
        single_label = torch.concat((single_label, single_lb.to(single_label.device)), dim = 0)
        
        return single_lb
        
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
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                #train_acc_per_class = acc_per_class(preds, labels, num_classes)

                wandb.log({
                "Train loss": train_loss,
                "Train acc" : train_acc,
                }) #, step = idx)
                
                # for i in range(num_classes):
                #     wandb.log({
                #         "Class "+str(i): train_acc_per_class[i]
                #     })
                
                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_score_items = []
            
            #epoch 당 각 label의 acc를 계산하고 싶음 
            pred_mask_label = torch.tensor([])
            pred_gen_label = torch.tensor([])
            pred_age_label = torch.tensor([])
            gt_mask_label = torch.tensor([])
            gt_gen_label = torch.tensor([])
            gt_age_label = torch.tensor([])
                      
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                
                
                f1_score_item = multiclass_f1_score(preds, labels , 
                                               num_classes=num_classes, average="macro").item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_score_items.append(f1_score_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                    
                    pred_mask_label, pred_gen_label, pred_age_label = concat_label(preds, pred_mask_label, pred_gen_label, pred_age_label)
                    gt_mask_label, gt_gen_label, gt_age_label = concat_label(labels, gt_mask_label, gt_gen_label, gt_age_label)
                                
            val_f1_score = np.sum(val_f1_score_items) / len(val_loader)
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} ||"
                f"f1_Score: {val_f1_score:4.2%}"
            )
            #val_acc_per_class = acc_per_class(preds, labels, num_classes)
                
            logger.add_scalar("Val/loss", val_loss, epoch) 
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            wandb.log({
                "Val loss":  val_loss,
                "Val acc" : val_acc,
                "Val f1_score": val_f1_score
            })

            val_mask_acc = acc_per_class(pred_mask_label, gt_mask_label, 3)
            val_gen_acc = acc_per_class(pred_gen_label, gt_gen_label, 2)
            val_age_acc= acc_per_class(pred_age_label, gt_age_label, 3)

            wandb.log({
                "(Mask) MASK ": val_mask_acc[0],
                "(Mask) INCORRECT ": val_mask_acc[1],
                "(Mask) NORMAL ": val_mask_acc[2],
                
                "(Gender) MALE ": val_gen_acc[0],
                "(Gender) FEMALE ": val_gen_acc[1],
                
                "(Age) YOUNG ": val_age_acc[0],
                "(Age) MIDDLE ": val_age_acc[1],
                "(Age) OLD ": val_age_acc[2],
            })

            
            # for i in range(num_classes):
            #     wandb.log({
            #             "Class "+str(i): val_acc_per_class[i]
            #     })
                
                
        #wandb.finish()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default=folder_name, help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/results'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)


#python3 custom_train.py --model 'MobileNet' --name 'MobileNet_ageonly_uppercrop_ce' --augmentation 'UpperFaceCropAugmentation' --epoch 30