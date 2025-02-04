from typing import Literal

from pydantic import BaseModel
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS

PL_STRATEGY = Literal[
    "ddp", "ddp_find_unused_parameters_true", 
    "deepspeed_stage_2", "deepspeed_stage_2_offload",
    "deepspeed_stage_3", "deepspeed_stage_3_offload",
]
TRAINSET = Literal['coco', 'smartphone']

class Config(BaseModel):
    wandb_project_name: str
    wandb_run_name: str
    wandb_host: str
    wandb_key: str

    pl_strategy: PL_STRATEGY
    pl_precision: _PRECISION_INPUT_STR | _PRECISION_INPUT_STR_ALIAS

    teacher_model: str
    student_model: str

    num_epochs: int
    student_lr: float
    student_lr_scheduler: str
    trainset: TRAINSET
    batch_size: int
    dataloader_num_worker: int


    # distillation
    distillation_loss_type: str
    temperature: float
    alpha: float

    train_all_params: bool

    projector_type: str


class SSLConfig(BaseModel):
    wandb_project_name: str
    wandb_run_name: str
    wandb_host: str
    wandb_key: str

    pl_strategy: PL_STRATEGY
    pl_precision: _PRECISION_INPUT_STR | _PRECISION_INPUT_STR_ALIAS

    model: str
    ssl_type: str
    
    num_epochs: int
    lr: float
    trainset: str
    batch_size: int
    dataloader_num_worker: int

    input_size: int