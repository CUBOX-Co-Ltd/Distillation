import torch
import torch.nn as nn

class KnowledgeDistiler:
    def __init__(
        self,
        fabric,
        config,
        trainset,
        trainloader
    ):
        super().__init__()
        self.fabric = fabric
        self.cfg = config
        self.logger: WandbLogger = self.fabric.logger

        self.device = self.fabric.device
        self.init_weight_dtype()

        self.init_dataloader(trainset, trainloader)
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

    def init_dataloader(self, trainset=None, trainloader=None):
        if trainloader is None:
            trainloader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=self.cfg.batch_size, 
                shuffle=True, 
                num_workers=self.cfg.dataloader_num_worker,
                collate_fn=getattr(trainset, "collate_fn", None),)

        print('trainloader', type(trainloader))
        self.trainloader = self.fabric.setup_dataloaders(trainloader)

    def init_model_and_optimizer(self, ):
        self.init_model()
        self.set_model_requires_grad()
        self.init_optimizer()
        self.fabric_setup()

    def init_model(self, ):
        pass
    
    def set_model_requires_grad(self, ):
        pass

    def init_optimizer(self,):
        pass

    def fabric_setup(self,):
        pass