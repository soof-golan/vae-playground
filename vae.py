# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deps

# %%
# Update Pip
# %pip install --quiet -U pip

# Install deps
# %pip install wandb protobuf==3.20 tqdm lightning ipywidgets torchmetrics optuna optuna-dashboard

# Update to pytorch 2.x
# %pip install -U "torch>=2.0,<3" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# %% [markdown]
# # Sanity
#
# If this fails, make sure you run on a device with an NVIDIA GPU

# %%
# !nvidia-smi

# %% [markdown]
# # Login to W&B

# %%
import wandb
wandb.login()

# %% [markdown]
# # Housekeeping 🏠🧹

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# Training
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from torchmetrics.functional.classification import accuracy

# Logging
import wandb

# Hyper-Parameter Search
import optuna
from optuna_lightning_helper import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback

# %% [markdown]
# # Configuration
#
# | Parameter  | Description | 
# | ---------- | ------------|
# | `BATCH_SIZE`                                    | Make as big as you can fit in VRAM |
# | `torch.set_float32_matmul_precision('medium')`  | To speed up comutations |
# | `PATH_DATASETS` | Where to store the datasets on disk (TODO: maybe use `ramfs`) |
# | `OPTUNA_DATABASE_URL` | Where to store hyper-parameter search progress (this is resumable keep this file handy) |

# %%
BATCH_SIZE = 2048
torch.set_float32_matmul_precision('medium')
PATH_DATASETS = "data"
OPTUNA_DATABASE_URL = "sqlite:///db.sqlite3"
PROJECT_NAME = "soof-autoencoder-v6"

dict(
    PROJECT_NAME=PROJECT_NAME,
    BATCH_SIZE=BATCH_SIZE,
    OPTUNA_DATABASE_URL=OPTUNA_DATABASE_URL,
    PATH_DATASETS=PATH_DATASETS,
)


# %% [markdown]
# # Load dataset 📦📦📦
#
# A [LightningDataModule][dm] that wraps the [FashionMNIST][ds] dataset
#
# ![Fashion MNIST sprite sheet][sprite]
#
# [sprite]: https://github.com/zalandoresearch/fashion-mnist/blob/c29cd591aa1b867e4b59227ee9a08ed0d4d4b34d/doc/img/fashion-mnist-sprite.png?raw=true
# [dm]: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
# [ds]: https://github.com/zalandoresearch/fashion-mnist
#

# %% [markdown]
# # Transforms
#
# During training the data is transformed with:
#
# * Normalization - standardizing the mean and std of the samples
# * TODO: Salt-and-peper noise
# * TODO: Gaussian noise
# * TODO: Blurs
# * TODO: Crops
# * TODO: Shifts
# * TODO: Warps

# %%
class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = 512, **kw):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # See transforms in the docs above
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # Magic numbers from FashionMNIST
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # Download the dataset
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(mnist_full, [59000, 1000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, pin_memory=True, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True, num_workers=0)
    
    def sample_train(self, num: int):
        self.train_data


# %%
class MLPEncoder(nn.Module):
    def __init__(self, size: int = 28, latent_dim: int = 16, act_fn: object = nn.GELU, **kw):
        super().__init__()
        self.size = size
        factors = [2, 4, 8]
        self.seq = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(in_features=size*size, out_features=latent_dim * factors[2]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[2], out_features=latent_dim * factors[1]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[1], out_features=latent_dim * factors[0]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[0], out_features=latent_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
        
class MLPDecoder(nn.Sequential):
    def __init__(self, size: int = 28, latent_dim: int = 16, act_fn: object = nn.GELU, **kw):
        super().__init__()
        self.size = size
        factors = [2, 4, 8]
        self.seq = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim * factors[0]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[0], out_features=latent_dim * factors[1]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[1], out_features=latent_dim * factors[2]), act_fn(),
            nn.Linear(in_features=latent_dim * factors[2], out_features=size*size), nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *rest = x.shape
        return self.seq(x).view(b, c, self.size, self.size)

class MLPAutoencoder(nn.Module):
    def __init__(self, size: int = 28, latent_dim: int = 16, act_fn: object = nn.GELU, **kw):
        super().__init__()
        self.encoder = MLPEncoder(size=size, latent_dim=latent_dim, act_fn=act_fn, **kw)
        self.decoder = MLPDecoder(size=size, latent_dim=latent_dim, act_fn=act_fn, **kw)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
        
assert MLPAutoencoder()(torch.rand((2,1,28,28))).shape == (2,1,28,28)


# %% [markdown]
# # Model
#
# This is a convolution based auto-encoder
#
# [Excalidraw](https://excalidraw.com/#json=s5Oy3xAQNIlHolsjPFct0,QyBB8WzaIzGBG1Q5IeD1uw)
#
# ![](docs/auto-encoder.png)
#
#

# %%
class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 28x28 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 14x14 => 7x7
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 7x7 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
    


# %%
class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            
            # 4x4 => 7x7
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid, kernel_size=3, 
                output_padding=0,          # NOTE! This was modified to support 28x28 images, instead of 32x32
                padding=1, 
                stride=2
            ),  
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            
            # 7x7 => 14x14
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            
            # 14x14 => 28x28
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


# %%
class Autoencoder(nn.Module):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

        self.width = width
        self.height = height
        self.num_input_channels = num_input_channels

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



# %%
def create_model(*a,**kw):
    return MLPAutoencoder()
    return Autoencoder(
            num_input_channels=1,
            width=28,
            height=28,
            base_channel_size=32,  # TODO: hyperparam?
            latent_dim=64,  # TODO: hyperparam?
        )

assert isinstance(create_model(), nn.Module), f"Expected nn.Module got {type(create_model())}"


# %% [markdown]
# # Lightning Model Setup

# %%
class LightningAutoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        model = create_model("autoencoder")
        self.model = model
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(1, 1, 28, 28)

    def forward(self, x):
        return self.model(x)
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        
        TODO: We probably can get fancier than MSE, if we think this will help
        """
        x, _ = batch  # We do not need the labels
        b, c, h, w = x.shape
        x_hat = self.forward(x)
        
        x = x.view(b, -1)
        x_hat = x_hat.view(b, -1)
        loss = F.mse_loss(x, x_hat, reduction="mean")
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        loss = self._get_reconstruction_loss(batch)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            # self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


# %% [markdown]
# # Training Loop (Managed by Lightning)

# %%
def train(trial: optuna.trial.Trial, hyperparameters: dict):
    model = LightningAutoencoder(**hyperparameters)
    dm = FashionMNISTDataModule(**hyperparameters)
    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=10,
        enable_model_summary=False,
        accelerator="auto",
        logger=WandbLogger(
            project=PROJECT_NAME, 
            name=PROJECT_NAME,
        ),
        callbacks=[
            L.pytorch.callbacks.RichModelSummary(max_depth=5),
            LearningRateMonitor(logging_interval="step"), 
            TQDMProgressBar(refresh_rate=1,),
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        precision="bf16",
        log_every_n_steps=5, 
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dm)    
    return trainer.callback_metrics["val_loss"].item()


# %% [markdown]
# # Hyper-parameter Search (Managed by Optuna)

# %%
def objective(trial: optuna.trial.Trial) -> float:
    # Generate hyperparameters
    batch_size = 2048
    momentum = trial.suggest_float("momentum", 0.0, 1.0) # 0.9 Worked well
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) # 1e-2 Worked well
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True) # 5e-4 Worked well
    
    hyperparameters = dict(momentum=momentum, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)
    
    return train(trial, hyperparameters)

def run():
    wandb_kwargs = dict(
        project=PROJECT_NAME,
        name=PROJECT_NAME,
        save_code=True,
    )
    wandbc = WeightsAndBiasesCallback(
        metric_name="val_loss",
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    study = optuna.create_study(
        storage=OPTUNA_DATABASE_URL,            # Store run data here. Can be also postgres / mysql
        load_if_exists=True,                    # Allow Resuming
        study_name=PROJECT_NAME,
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner()    # Prune unpromising runs (early stopping)
    )
    study.optimize(
        objective, 
        n_trials=1,
        callbacks=[wandbc],
    )

    print("Number of finished trials: {}".format(len(study.trials)))
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
run()

# %%

# %%
