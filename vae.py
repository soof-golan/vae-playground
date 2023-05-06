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
# %pip install -U wandb==0.15.0 protobuf==3.20 tqdm lightning ipywidgets torchmetrics timm optuna optuna-dashboard

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
import lightning as L
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchmetrics.functional.classification import accuracy
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from torch.utils.data import DataLoader, random_split
import wandb
import optuna
from optuna_lightning_helper import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback

torch.set_float32_matmul_precision('medium')

PATH_DATASETS = os.environ.get("PATH_DATASETS", "data")
COMPILE_MODEL = os.environ.get("COMPILE_MODEL") and torch.cuda.is_available()
JIT_MODEL = os.environ.get("JIT_MODEL") and torch.backends.mps.is_available()
BATCH_SIZE = 2048
USE_MP_LOADER = os.environ.get("USE_MP_LOADERS")
DATABASE_URL =os.environ.get("DATABASE_URL", "sqlite:///db.sqlite3")
NUM_WORKERS = 0 if not USE_MP_LOADER else os.cpu_count()

print(f"""
Running Config:
{PATH_DATASETS=},
{COMPILE_MODEL=},
{BATCH_SIZE=},
{JIT_MODEL=},
{USE_MP_LOADER=},
""")


# %% [markdown]
# # Load dataset 📦📦📦

# %%
class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = 512, **kw):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

# assert (dm.prepare_data() is None or os.path.isdir("./data")), "Data Failed to load"
    


# %% [markdown]
# # Create Model

# %%
from timm.models.resnet import resnet18
from resnet import mono_resnet18


# %%
def create_model(model_type: str = "torch-modded", num_classes=10):
    if model_type == "timm":
        model = resnet18(pretrained=False, num_classes=num_classes, in_chans=1)
    elif model_type == "torch-modded":
        model = mono_resnet18(weights=None, num_classes=num_classes)
    else:
        raise ValueError("Invalid model")
    return model

assert isinstance(create_model(), nn.Module), f"Expected nn.Module got {type(create_model())}"
# create_model()

# %% [markdown]
# # Lightning Model Setup

# %%
class LitModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.example_input_array = torch.rand(BATCH_SIZE, 1, 28, 28)
        model = create_model()
        if JIT_MODEL:
            self.model = torch.jit.script(model)
        else:
            self.model = model
        self.save_hyperparameters()
        
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        logits = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        out = self(x)
        logits = F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

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
    model = LitModel(**hyperparameters)
    dm = FashionMNISTDataModule(**hyperparameters)
    if COMPILE_MODEL:
        model = torch.compile(model)
    wandb.init(project="soof-vea",name=f"optunda-{model.model.__class__.__name__}", save_code=True)
    logger = WandbLogger()

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="auto",
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"), 
            TQDMProgressBar(refresh_rate=1,),
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        ],
        precision="16-mixed",
        log_every_n_steps=5, 
    )
    trainer.logger.log_hyperparams(hyperparameters)
    try:
        trainer.fit(model, datamodule=dm)
        return trainer
    finally:
        wandb.finish()

# train()


# %% [markdown]
# # Hyper-parameter Search (Managed by Optuna)

# %%
def objective(trial: optuna.trial.Trial) -> float:
    # Generate hyperparameters
    batch_size = 1 << trial.suggest_int("batch_size", 6, 11)
    momentum = trial.suggest_float("momentum", 0.0, 1.0) # 0.9 Worked well
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) # 1e-2 Worked well
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True) # 5e-4 Worked well
    
    hyperparameters = dict(momentum=momentum, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)
    
    # Train
    trainer = train(trial, hyperparameters)
    
    # Return objective metric
    return trainer.callback_metrics["val_acc"].item()


def run():
    # wandb_kwargs = dict(project="soof-vea", save_code=True)
    # wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(
        storage=DATABASE_URL,  # Store run data here. Can be also postgres / mysql
        load_if_exists=True,             # Allow Resuming
        study_name="vea-cls-3", 
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner() # Prune unpromising runs (early stopping)
    )
    study.optimize(
        objective, 
        n_trials=100,
        # callbacks=[wandbc],
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
