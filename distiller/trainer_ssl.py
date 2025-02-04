import torch
import torch.nn as nn

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data.dataset import YOLODataset

from lightly.loss import NTXentLoss
from lightly.transforms import SimCLRTransform
from lightly.models.modules import SimCLRProjectionHead

from .util import default

class SimCLRTrainer:
    def __init__(
        self,
        fabric,
        config,
        model,
        trainset=None
    ):
        super().__init__()
        self.fabric = fabric
        self.cfg = config
        self.logger: WandbLogger = self.fabric.logger

        self.device = self.fabric.device

        self.model = model
        self.init_weight_dtype()

        self.init_dataloader(trainset)
        self.init_model_and_optimizer()

    def init_weight_dtype(self):
        precision_str = self.cfg.pl_precision

        if '16' in precision_str or 'transformer-engine' in precision_str:
            if 'bf' in precision_str:
                self.weight_dtype = torch.bfloat16
            else:
                self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

    def init_dataloader(self, trainset):
        transform = SimCLRTransform(input_size=self.cfg.input_size)

        if self.cfg.trainset == 'vocdet':
            trainset = torchvision.datasets.VOCDetection(
                "datasets/pascal_voc",
                download=True,
                transform=transform,
                target_transform=target_transform,
            )
        elif self.cfg.trainset == 'coco':
            data_yaml = '/purestorage/project/tyk/3_CUProjects/Distillation/ultralytics/cfg/datasets/coco.yaml'
            data = check_det_dataset(data_yaml)
            trainset = YOLODataset(
                data["train"],
                data=data,
                task="detect",
                imgsz=640,
                augment=False,
                batch_size=1,
            )
        elif trainset is not None:
            trainset = trainset
        else:
            trainset = None

        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.dataloader_num_worker,
            collate_fn=getattr(trainset, "collate_fn", None),)

        self.trainloader = self.fabric.setup_dataloaders(trainloader)

    def init_model_and_optimizer(self, ):
        self.init_model()
        self.set_model_requires_grad()
        self.init_optimizer()
        self.fabric_setup()

    def init_model(self, ):
        num_feats, self.backbone = self.setup_backbone(self.model)     
        self.projection_head = SimCLRProjectionHead(
            input_dim=num_feats,
            hidden_dim=num_feats,
            output_dim=128,
        )
        
        self.criterion = NTXentLoss(gather_distributed=True)

    def setup_backbone(self, model):
        backbone = self.model.backbone
        with torch.no_grad():
            out = backbone(torch.randn(1,3,640,640))
            print('out', out.shape)

        return out.shape[-1], backbone

    def set_model_requires_grad(self, ):
        pass

    def init_optimizer(self,):
        self.optimizer = torch.optim.SGD(
            [self.backbone.parameters()]+[self.projection_head.parameters()], 
            lr=default(self.cfg.lr, 0.06)
        )

    def fabric_setup(self,):
        self.backbone, self.optimizer = self.fabric.setup(model, optimizer)

    def train_step(self, batch):
        (x0, x1) = batch[0]
        z0 = self.projection_head(self.backbone(x0).flatten(start_dim=1))
        z1 = self.projection_head(self.backbone(x1).flatten(start_dim=1))
        loss = self.criterion(z0, z1)
        return loss

    def train(self,):
        for epoch in range(self.cfg.num_epochs):
            self.fabric.print(f"Epoch {epoch + 1}")
            total_loss = 0

            self.trainloader = tqdm(self.trainloader) if self.fabric.global_rank == 0 else self.trainloader
            for batch in self.trainloader:
                loss = self.train_step(batch)

                # Backpropagation
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

            torch.save(self.backbone.state_dict(), f'{self.cfg.pth_path}/{epoch}')

from lightly.transforms.dino_transform import DINOTransform
from lightly.models.modules import DINOProjectionHead
class DINOTrainer:
    def __init__(
        self,
        fabric,
        config,
        model,
    ):
        super().__init__()
        self.fabric = fabric
        self.cfg = config
        self.logger: WandbLogger = self.fabric.logger

        self.device = self.fabric.device

        self.model = model
        self.init_weight_dtype()

        self.init_dataloader()
        self.init_model_and_optimizer()

    def init_weight_dtype(self):
        precision_str = self.cfg.pl_precision

        if '16' in precision_str or 'transformer-engine' in precision_str:
            if 'bf' in precision_str:
                self.weight_dtype = torch.bfloat16
            else:
                self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

    def init_dataloader(self):
        transform = DINOTransform()
        def target_transform(t):
            return 0

        if self.cfg.trainset == 'vocdet':
            trainset = torchvision.datasets.VOCDetection(
                "datasets/pascal_voc",
                download=True,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            trainset = None

        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.dataloader_num_worker,
            collate_fn=getattr(trainset, "collate_fn", None),)

        self.trainloader = self.fabric.setup_dataloaders(trainloader)

    def init_model_and_optimizer(self, ):
        self.init_model()
        self.set_model_requires_grad()
        self.init_optimizer()
        self.fabric_setup() 

    def init_model(self, ):
        ...

    def train_step(self, batch):
        ...