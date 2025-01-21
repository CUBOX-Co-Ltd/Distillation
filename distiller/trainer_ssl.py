import torch
import torch.nn as nn

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
        transform = SimCLRTransform(input_size=input_size)

        if config.trainset == 'vocdet':
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
        num_feats, self.backbone = setup_backbone(self.model)     
        self.projection_head = SimCLRProjectionHead(
            input_dim=num_feats,
            hidden_dim=num_feats,
            output_dim=128,
        )
        
        self.criterion = NTXentLoss(gather_distributed=True)


    def set_model_requires_grad(self, ):
        pass

    def init_optimizer(self,):
        self.optimizer = torch.optim.SGD(
            [self.backbone.parameters()]+[self.projection_head.parameters()], 
            lr=default(self.config.lr, 0.06)
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
        for epoch in range(self.config.num_epochs):
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

        if config.trainset == 'vocdet':
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
        

    def train_step(self, batch):
