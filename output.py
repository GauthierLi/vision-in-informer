import os
import time
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vii_train import build_model, build_transform

import pandas as pd
import torchvision.transforms as T


class pred_dataset(Dataset):
    """imgdata with life label"""

    def __init__(self, folder_path, transformer=None):
        super().__init__()
        self.folder_path = folder_path
        self.imgs = list(os.listdir(self.folder_path))
        self.transformer = transformer

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.folder_path, self.imgs[index])

        img = Image.open(img_path)
        if self.transformer is not None:
            img = self.transformer(img)
        return img


def load_resume(CFG):
    model = CFG.model
    assert os.path.exists(CFG.resume), f"CHECKPOINT {CFG.resume} NOT EXIT!"
    state_dict = torch.load(os.path.join(os.getcwd(),CFG.resume))
    model.load_state_dict(state_dict)
    model.to(CFG.device)
    return model


def get_life_list(model, data_loader):
    life_list = []
    with torch.no_grad():
        bar = tqdm(data_loader)
        for img in bar:
            pred = model(img.to(CFG.device)).squeeze().cpu().detach().item()
            life_list.append(pred)
    return life_list

def list_to_time_stap(l1):
    list_with_stap = []
    for i in l1:
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        tmp = []
        tmp.append(now)
        tmp.append(i)
        list_with_stap.append(tmp)
        time.sleep(1)
    df = pd.DataFrame(list_with_stap, columns=["date","life"])
    return df

if __name__ == "__main__":
    class CFG:
        device = "cuda" if torch.cuda.is_available else "cpu"
        model = build_model()
        resume = r"checkpoints/vii_epochs30_bs16/best_epoch.pth"

        pred_imgs_path = "data/cwt_raw_data"

        save_csv = "test.csv"

    vii_model = load_resume(CFG=CFG)

    trans = build_transform()
    dataset = pred_dataset(CFG.pred_imgs_path, transformer=trans)
    data_loader = DataLoader(dataset=dataset, batch_size=1)

    life_predict = get_life_list(vii_model, data_loader)
    life_datafram = list_to_time_stap(life_predict)

    life_datafram.to_csv(CFG.save_csv, index=False)
    print(life_datafram)
