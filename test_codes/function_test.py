import torch 
import numpy as np

pred = torch.tensor([1,  6, 1,  0,  5,  0, 5])
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])

def acc_per_class(pred, labels, num_classes):

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
            class_acc[i] = np.round(labels_hit[i]/labels_per_class[i], 3)
    
    #print
    for i in class_acc:
        print(f"class {i} acc: {class_acc[i]}%")
    

print(acc_per_class(pred, labels, num_classes=15))
            
    

