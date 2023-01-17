from pathlib import Path
import argparse
import math
from functools import partial
from typing import Union
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import torch
from lightning_lite.utilities.seed import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb_params
from data_module import TinyImageNetDiffDataModule
from model import *
from image_utils import *


default_config = SimpleNamespace(
    download_data=True,
    debug_run=False,
    n_channels=3,
    image_size=64,
    batch_size=128,
    timesteps=1000,
    schedule='quadratic',
    augment=True,
    epochs=200,
    n_workers=1,
    gradient_clip_val=10.,
    early_stopping=False,
    patience=None,
    lr=0.005,
    seed=42,
    position_dim=64,
    cosine_period=100,
    n_test_images=49,
    animate_image_no=7,
    plot_every_n_steps=10,
    display_final_images=True,
    offline_data_dir="./data/processed",
    output_dir="./outputs",
)


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass


def parse_args():
    parser = argparse.ArgumentParser(description="Training and validation hyperparameters")
    parser.add_argument('--download_data', type=t_or_f, default=default_config.download_data, help=f"Set to 'true' "
                        f"to download the data from a WandB artifact")
    parser.add_argument('--debug_run', type=t_or_f, default=default_config.debug_run, help="Set to 'true' to disable WandB logging")
    parser.add_argument('--n_channels', type=int, default=default_config.n_channels, help="Num. channels in the images (RGB=3)")
    parser.add_argument('--image_size', type=int, default=default_config.image_size, help="Image size used to train the model")
    parser.add_argument('--batch_size', type=int, default=default_config.batch_size, help="Batch size used to train the model")
    parser.add_argument('--timesteps', type=int, default=default_config.timesteps, help="Number of timesteps in the diffusion")
    parser.add_argument('--schedule', type=str, default=default_config.schedule, help="Type of variance schedule; linear/quadratic/sigmoid")
    parser.add_argument('--augment', type=t_or_f, default=default_config.augment, help="Whether to apply data augmentation")
    parser.add_argument('--epochs', type=int, default=default_config.epochs, help="Number of epochs in the training run")
    parser.add_argument('--n_workers', type=int, default=default_config.n_workers, help="Number of workers to use in the dataloader")
    parser.add_argument('--gradient_clip_val', type=float, default=default_config.gradient_clip_val, help=f"Gradient clipping value to "
                        f"stabilise training")
    parser.add_argument('--early_stopping', type=t_or_f, default=default_config.early_stopping, help=f"Whether to stop the run early if "
                        f"val loss increases")
    parser.add_argument('--patience', type=int, default=default_config.patience, help="Patience (in epochs) for early stopping")
    parser.add_argument('--lr', type=float, default=default_config.lr, help="Initial learning rate")
    parser.add_argument('--seed', type=int, default=default_config.seed, help="Random seed to use for the run")
    parser.add_argument('--position_dim', type=int, default=default_config.position_dim, help="Num. of dimension of the position embedding")
    parser.add_argument('--cosine_period', type=int, default=default_config.cosine_period, help="Learning rate resets every N epochs")
    parser.add_argument('--n_test_images', type=int, default=default_config.n_test_images, help=f"Number of images to include in the FID "
                        f"score and animation")
    parser.add_argument('--animate_image_no', type=int, default=default_config.animate_image_no, help=f"Index of the image to "
                        f"create an animation from")
    parser.add_argument('--plot_every_n_steps', type=int, default=default_config.plot_every_n_steps, help=f"Include only this frequency "
                        f"of images in the animation")
    parser.add_argument('--display_final_images', type=t_or_f, default=default_config.display_final_images, help=f"Whether to "
                        f"display the final step of each image in the test batch")
    parser.add_argument('--offline_data_dir', type=str, default=default_config.offline_data_dir, help=f"Directory for offline data "
                        f"(ignored if not a debug run)")
    parser.add_argument('--output_dir', type=str, default=default_config.output_dir, help="Directory for output files")
    args = parser.parse_args()
    vars(default_config).update(vars(args))
    
    
def download_data(run: wandb.run, offline_data_dir = './data/processed'):
    if run.mode == 'online':
        processed_data_art = run.use_artifact(f"{wandb_params.PROCESSED_DATA_ART}:latest")
        processed_data_dir = processed_data_art.download()  # the files are downloaded in the artifacts directory by default.
        print(f"Data files downloaded in {processed_data_dir}.")
        return processed_data_dir

    else:
        return offline_data_dir
    

def get_df(processed_data_dir: Union[Path, str]):
    if isinstance(processed_data_dir, str):
        processed_data_dir = Path(processed_data_dir)
    df = pd.read_csv(processed_data_dir / "splits.csv")
    df['is_valid'] = df['split'] == "valid"
    df.drop('split', axis=1, inplace=True)    
    return df


def train(run, config):
    if config.download_data and (not config.debug_run):
        processed_data_dir = download_data(run)
    else:
        processed_data_dir = config.offline_data_dir
    df = get_df(processed_data_dir)
    
    dmodule = TinyImageNetDiffDataModule(
        filelist_df=df,
        batch_size=config.batch_size,
        timesteps=config.timesteps,
        train_transforms=train_transforms if config.augment else fwd_fromPIL_transforms,
        val_transforms=fwd_fromPIL_transforms,
        num_workers=config.n_workers
    )
    
    callbacks = [LearningRateMonitor('epoch')]
    if config.early_stopping:
        callbacks.append(EarlyStopping('val_loss', patience=config.patience)) 
    if run.mode != 'disabled':
        logger = WandbLogger(
            project=wandb_params.WANDB_PROJECT, log_model=True, name=wandb_params.TRAINING_RUN_NAME
        )
    else:
        logger = None
        
    trainer = pl.Trainer(
        accelerator='gpu', 
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=".",
        enable_model_summary=True,
        callbacks=callbacks, 
        strategy='dp',
        max_epochs=config.epochs,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm='value', 
    )
    
    model = UNetDiffModule(
        time_emb_dim=config.position_dim,
        lr=config.lr,
        n_channels=config.n_channels, 
        cosine_period=config.cosine_period
    )
    model.to('cuda')
    trainer.fit(model, dmodule)
    return trainer, model, dmodule
    

def test(trainer, model, dmodule, config, return_trajectory = True):
    
    test_results = trainer.test(model, dmodule)
    test_loss = test_results[0]['test_loss']
    
    trajectory = sample_trajectory(
        config.timesteps,
        config.schedule,
        1e-4,
        0.02,
        config.n_test_images,
        (config.image_size, config.image_size),
        config.n_channels, 
        model, 
        'cuda', 
        None
    )
    
    fid = FID(dmodule, clip_values=False)
    fid.setup()
    fid_score = fid.compute(trajectory[-1])
    
    if return_trajectory:
        return test_loss, fid_score, trajectory
    else:
        return test_loss, fid_score
    

def make_animation(trajectory, index_in_batch, plot_every_n_steps, save_dir = "./outputs"):
    f = plt.figure(figsize=(4, 4))
    images = list()
    for i in tqdm(range(len(trajectory))):
        if i % plot_every_n_steps == 0 or i == len(trajectory) - 1:
            img = bwd_toNumpy_transforms(trajectory[i][index_in_batch])
        images.append([plt.imshow(img)])

    animation = anim.ArtistAnimation(f, images, interval=50, blit=True, repeat_delay=1000)
    animation.save(save_dir + '/diffusion.gif', writer='pillow', fps=30, dpi=50)
    print("Reverse diffusion animation saved.")
    

def get_final_images(trajectory, save_dir = "./outputs"):
    n_rows = math.ceil(math.sqrt(trajectory[-1].shape[0]))
    size = n_rows * 2
    show(
        list(map(bwd_toNumpy_transforms, trajectory[-1])),
        figsize=(size, size), 
        nrows=n_rows, 
        save_dir=save_dir
    )
    print("Final images saved.")    
    

def main(config):
    seed_everything(config.seed, workers=True)
    mode = 'offline' if config.debug_run else 'online'
    run = wandb.init(project=wandb_params.WANDB_PROJECT, job_type="training", config=config, mode=mode)
    trainer, model, dmodule = train(run, config)
    if config.animate_image_no is not None:
        test_loss, fid_score, trajectory = test(trainer, model, dmodule, config, True)
        assert trajectory[-1].shape[0] > config.animate_image_no, \
            "You asked for an image index that exceeds the number of images."
        make_animation(trajectory, config.animate_image_no, config.plot_every_n_steps, config.output_dir)
    else:
        test_loss, fid_score = test(trainer, model, dmodule, config, False)
    
    get_final_images(trajectory, save_dir=config.output_dir)

    output_art = wandb.Artifact(name=wandb_params.OUTPUT_ART, type="output")
    output_art.add_dir(config.output_dir)
    wandb.log_artifact(output_art)

    print(f"Training completed; the model achieves a validation loss of {test_loss : .4f} and "
          f"a FID score of {fid_score : .2f}.")
    
    wandb.summary['test_loss'] = test_loss
    wandb.summary['fid_score'] = fid_score    
    wandb.finish()


if __name__ == "__main__":
    parse_args()
    main(default_config)