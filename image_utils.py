
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


class PermuteChannels():
    def __init__(self, mode = 'last'):
        self.mode = mode
        
    def __call__(self, t):
        if self.mode == 'first':
            if len(t.shape) < 3:
                t = t.unsqueeze(-1)
            return t.permute(2, 0, 1) if isinstance(t, torch.Tensor) else t.transpose((2, 0, 1))
    
        if self.mode == 'last':
            if len(t.shape) < 3:
                t = t.unsqueeze(0)
            return t.permute(1, 2, 0) if isinstance(t, torch.Tensor) else t.transpose((1, 2, 0))
        
        raise ValueError(f"`{self.mode}` is not a valid permutation mode.")


class ValueScaler():
    def __init__(self, source_scale = '0_255', target_scale = '-1_1', to_numpy = False):
        """
        source_scale: str, any of '0_255', '-1_1', '0_1', 'unbound'. Default = '0_255'
        target_scale: str, same values. Default = '-1_1'
        """

        assert source_scale in ['0_255', '-1_1', '0_1', 'unbound']
        assert target_scale in ['0_255', '-1_1', '0_1']
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.to_numpy = to_numpy

    def __call__(self, t):
        if self.source_scale == self.target_scale:
            out = t

        elif self.source_scale == '0_1':
            if self.target_scale == '0_255':
                out = t * 255
            if self.target_scale == '-1_1':
                out = t * 2. - 1

        elif self.source_scale == '-1_1':
            if self.target_scale == '0_255':
                out = t * 127.5 + 127.5
            if self.target_scale == '0_1':
                out = t * 0.5 + 0.5

        elif self.source_scale == '0_255':
            if self.target_scale == '0_1':
                out = t / 255.
            if self.target_scale == '-1_1':
                out = t * 2 / 255. - 1

        elif self.source_scale == 'unbound':
            out = (t - t.min()) / (t.max() - t.min())
            out = ValueScaler(source_scale='0_1', target_scale=self.target_scale)(out)
        else:
            raise ValueError("Either source or target scales are incorrect (possibly both)")

        return out.cpu().numpy() if self.to_numpy else out


fwd_fromPIL_transforms =  transforms.Compose(
        [
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            ValueScaler(source_scale='0_1', target_scale='-1_1')  # ToTensor performs [0-1] scaling implicitly
        ]
    )

train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=(64, 64), 
                padding=5,
                pad_if_needed=True,
                padding_mode='edge'
            ),
            transforms.RandomAutocontrast(),
            fwd_fromPIL_transforms,
        ]
    )

fwd_fromNumpy_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            PermuteChannels(mode='first'),
            transforms.CenterCrop((64, 64)),
            ValueScaler(source_scale='0_1', target_scale='-1_1'),
        ]
    )

bwd_toPIL_transforms = transforms.Compose(
        [
            ValueScaler(source_scale='unbound', target_scale='0_255', to_numpy=False),
            transforms.ToPILImage(mode='RGB')
        ]
    )

bwd_toNumpy_transforms = transforms.Compose(
        [
            ValueScaler(source_scale='unbound', target_scale='0_1', to_numpy=True),
            PermuteChannels(mode='last'),
        ]
    )


def make_noisy_image(initial_image, step, schedule, beta_min, beta_max, timesteps):
    coeffs = make_coeffs(schedule, beta_min, beta_max, timesteps)
    if isinstance(initial_image, list) and len(intial_image) == 2:
        intial_image = initial_image[0]
    img = initial_image if initial_image.dim() == 3 else initial_image.squeeze()
    return bwd_toNumpy_transforms(noisify(img, t=torch.tensor([step]), coeffs=coeffs))


def plot_steps(initial_image, steps, schedule, beta_min, beta_max, timesteps):
    if initial_image.dim() > 3:
        initial_image = initial_image.squeeze()
    images = [bwd_toNumpy_transforms(initial_image)] + [
        make_noisy_image(initial_image, step, schedule, beta_min, beta_max, timesteps)
        for step in steps
    ]
    return images


def show(imgs, figsize = None, nrows = 1, save_dir = None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    ncols = ceil(len(imgs) / nrows)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=figsize)
    fig.tight_layout()
    for i, img in enumerate(imgs):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(np.array(img))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.close(fig)
    if save_dir is not None:
        fig.savefig(save_dir + '/final_images.png', dpi=fig.dpi, bbox_inches='tight')
