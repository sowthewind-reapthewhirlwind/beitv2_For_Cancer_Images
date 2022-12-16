import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
def build_dataset_for_evaluation_confusion_matrix(path,model):
    batch_size=24
    y_pred = []
    y_true = []
    transform = transforms.Compose(
    [transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(root=path,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('cancer','normal')
    i=0
    for inputs, labels in testloader:
        if i>100:
            break
        i+=1
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *2, index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("plt.png")