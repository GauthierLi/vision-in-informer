#############################################################################################
### @author: gauthierLi                                                                  ####
### @email: lwklxh@163.com                                                               ####
### @date: 22/06/22                                                                      ####
### change configuration by changing CFG class, and use                                  ####
###         "from vii_model import VisionTransformer" to use vii model,                  ####
###         "from vit_model import VisionTransformer" to use vit model,                  ####
### parently can found that vit model performence better than vii from training result.  ####
#############################################################################################
import os
import sys
import pdb
import math
import time
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from vii_model import VisionTransformer

import torchvision.transforms as T


################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> data loader and transform <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class vii_dataset(Dataset):
    """imgdata with life label"""

    def __init__(self, folder_path, transformer=None):
        super().__init__()
        self.folder_path = folder_path
        self.imgs = list(os.listdir(self.folder_path))
        self.transformer = transformer

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_name = self.imgs[index]
        _, cur_life, total_life = img_name.split(".")[0].split("_")
        cur_life, total_life = float(cur_life), float(total_life)
        img_path = os.path.join(self.folder_path, self.imgs[index])

        label = round(cur_life / total_life, 5)

        img = Image.open(img_path)
        if self.transformer is not None:
            img = self.transformer(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def build_transform():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5], [0.5])])


def build_dataset(CFG, transform=None):
    train_dataset = vii_dataset(CFG.train_fold, transformer=transform)
    val_dataset = vii_dataset(CFG.val_fold, transformer=transform)
    return train_dataset, val_dataset


def build_dataloader(CFG, transform):
    train_data_set, val_data_set = build_dataset(CFG, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=CFG.batch_size,
                                                    shuffle=True,
                                                    num_workers=CFG.num_workers,
                                                    pin_memory=False,
                                                    collate_fn=train_data_set.collate_fn
                                                    )
    val_data_loader = torch.utils.data.DataLoader(val_data_set,
                                                  batch_size=CFG.batch_size,
                                                  shuffle=True,
                                                  num_workers=CFG.num_workers,
                                                  pin_memory=False,
                                                  collate_fn=val_data_set.collate_fn
                                                  )
    return train_data_loader, val_data_loader

############################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> build  model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def build_model():
    return VisionTransformer(img_size=224,
                             patch_size=16,
                             embed_dim=768,
                             depth=12,
                             num_heads=12,
                             representation_size=768,
                             num_classes=1)

#############################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> build loss <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def build_loss():
    return nn.MSELoss()

#############################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>> train & valid and test one epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def train_one_epoch(CFG, train_data_loader, optimizer, model, epoch):
    model.to(CFG.device)
    model.train()
    loss = build_loss()
    optimizer.zero_grad()

    loss_all = 0
    data_loader = tqdm(train_data_loader, file=sys.stdout)
    for step, (img, label) in enumerate(data_loader):
        pred = model(img.to(CFG.device)).squeeze(dim=1)

        mse_loss = loss(pred, label.to(CFG.device))
        mse_loss.backward()
 
        loss_all += mse_loss.item()
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, loss_all / (step + 1))
        optimizer.step()
        optimizer.zero_grad()
    
    return loss_all / (step + 1)

@torch.no_grad()
def val_one_epoch(CFG, val_dataloader, model, epoch):
    model.to(CFG.device)
    metric = build_loss()

    val_loader = tqdm(val_dataloader, file=sys.stdout)
    val_all = 0
    for step, (img, label) in enumerate(val_loader):
        pred = model(img.to(CFG.device)).squeeze(dim=1)
        mse_loss = metric(pred, label.to(CFG.device))
        
        val_all += mse_loss
        val_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, val_all / (step + 1))
    return val_all / (step + 1)

@torch.no_grad()
def test_one_epoch(CFG, model, export_to_file = True):
    model.to(CFG.device)
    best_epoch = os.path.join(CFG.save_fold, "best_epoch.pth")
    state_dict = torch.load(best_epoch)
    model.load_state_dict(state_dict)
    result = [[],[]]

    transform = build_transform()
    test_dataset = vii_dataset(CFG.test_fold, transformer=transform)

    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=CFG.batch_size,
                                                  shuffle=False,
                                                  num_workers=CFG.num_workers,
                                                  pin_memory=False,
                                                  collate_fn=test_dataset.collate_fn
                                                  )

    data_loader = tqdm(test_data_loader, file=sys.stdout, desc="[testing]")
    for step, (img,  label) in enumerate(data_loader):
        pred = model(img.to(CFG.device)).squeeze(dim=1)
        result[0] += pred.cpu().numpy().tolist()
        result[1] += label.cpu().numpy().tolist() 

    df = pd.DataFrame({"predict":result[0], "ground truth":result[1]})
    if export_to_file:
        df.to_csv("test_result.csv")
    return df
############################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def mkfolder(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(name=folder_name)


if __name__ == '__main__':
    class CFG:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 16
        num_workers = 2
        epochs = 30

        lr = 1E-3
        lrf = 1E-2
        w_decay = 5E-5

        train_fold = r"/media/gauthierli-org/GauLi/code/王总项目/vision_transformer/data/train_data"
        val_fold = r"/media/gauthierli-org/GauLi/code/王总项目/vision_transformer/data/valid_data"
        test_fold = r"/media/gauthierli-org/GauLi/code/王总项目/vision_transformer/data/valid_data"
        save_fold = "checkpoints/vii_epochs{}_bs{}".format(epochs, batch_size)

    transform = build_transform()
    train_dataloader, val_dataloader = build_dataloader(CFG=CFG, transform=transform)
    model = build_model()


#################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> train process <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    train_flag = True
    if train_flag:
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=CFG.lr, momentum=0.9, weight_decay=CFG.w_decay)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / CFG.epochs)) / 2) * (1 - CFG.lrf) + CFG.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        best = 99999
        for epoch in range(CFG.epochs):
            start_time = time.time()
            train_loss = train_one_epoch(CFG=CFG, train_data_loader=train_dataloader,
                                    optimizer=optimizer,model=model, epoch=epoch)
            val_loss = val_one_epoch(CFG=CFG, val_dataloader=val_dataloader, model=model, epoch=epoch)
            end_time = time.time()

            print("[{} epoch]: Train loss : {:.4f} , valid loss : {:.4f}, use time {:.2f}s. \n".format(
                                        epoch, train_loss, val_loss, end_time - start_time), flush=True)


            if not os.path.isdir(CFG.save_fold):
                mkfolder(CFG.save_fold)
            if val_loss < best:
                best = val_loss
                ckpt_path = os.path.join(CFG.save_fold, f"best_epoch.pth")
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                torch.save(model.state_dict(), ckpt_path)
                print("saving best ... ...", flush=True)

            last_epoch = os.path.join(CFG.save_fold, "epoch_{}.pth".format(epoch))
            torch.save({"state_dict": model.state_dict, "epoch":epoch}, last_epoch)
            print("saving last ... ...", flush=True)
            if os.path.isfile(os.path.join(CFG.save_fold, "epoch_{}.pth".format(epoch - 1))):
                os.remove(os.path.join(CFG.save_fold, "epoch_{}.pth".format(epoch - 1)))

        
##################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> test process <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    test_flag = True
    if test_flag:
        df = test_one_epoch(CFG, model,export_to_file=False)
        plt.scatter(df["predict"], df["ground truth"],marker=".")
        plt.xlabel("pred")
        plt.ylabel("gt")
        plt.show()
        