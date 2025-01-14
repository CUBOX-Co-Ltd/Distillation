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
        trainloader=None
    ):
        super().__init__(fabric, config, trainset, trainloader)


    def init_model(self, teacher, student):
        self.aligner = nn.ModuleDict()


        for stage in 

