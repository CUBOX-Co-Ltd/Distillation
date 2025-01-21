import torch

from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from ultralytics import YOLO

import importlib
from tqdm import tqdm

from .distil_common import BaseDistiller
from .util import preprocess_batch
from .loss import BasicDistillationLoss


class OFADistiler(BaseDistiller):
    def __init__(
        self,
        fabric,
        config,
        trainset,
        teacher,
        student,
        trainloader=None
    ):
        super().__init__(fabric, config, trainset, teacher, student, trainloader)


    def init_model(self, teacher, student):
        self.aligner = nn.ModuleDict()
        # for stage in 

    def set_model_requires_grad(self, ):
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        if self.cfg.train_all_params:
            for param in self.student.parameters():
                param.requires_grad = True
        else:
            pass