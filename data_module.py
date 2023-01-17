
from typing import Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import pytorch_lightning as pl

from image_utils import fwd_fromPIL_transforms


class TinyImageNetDiffInMem(dset.VisionDataset):
    # Slightly faster implementation that loads the whole data in memory.
    
    def __init__(
        self, 
        filelist_df: pd.DataFrame,
        train = True,
        image_size = (64, 64),
        timesteps= 1000,
        img_transforms: Callable = fwd_fromPIL_transforms,
    ):  
        assert filelist_df.columns.to_list() == ['filename', 'is_valid'], filelist_df.columns.to_list() 
        assert img_transforms is not None
        
        if train:
            self.filelist_df = filelist_df[~filelist_df.is_valid]
        else:
            self.filelist_df = filelist_df[filelist_df.is_valid]
            
        self.images = [Image.open(f).convert("RGB") for f in self.filelist_df['filename']]
        self.timesteps = timesteps
        self.img_transforms = img_transforms
        
    def __len__(self):
        return len(self.filelist_df)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.img_transforms(img)
        t = torch.randint(0, self.timesteps, (1,)).to(torch.float32)
        return img, t

    
class TinyImageNetDiffDataModule(pl.LightningDataModule):
    def __init__(self, filelist_df, batch_size, timesteps, train_transforms, val_transforms, num_workers):
        super().__init__()
        self.filelist_df = filelist_df
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_workers = num_workers
                
    def setup(self, stage = None):
        self.train = TinyImageNetDiffInMem(
            self.filelist_df,
            train=True,
            timesteps=self.timesteps,
            img_transforms=self.train_transforms,
        )
        self.val = TinyImageNetDiffInMem(
            self.filelist_df,
            train=False,
            timesteps=self.timesteps,
            img_transforms=self.val_transforms,
        )
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):  
        '''
        We don't need a separate test set because we are reconstructing images
        from random noise, not from existing ones. But defining this dataloader
        allows us to use pl.Trainer's test() method to measure the final model's 
        loss.
        '''
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)
