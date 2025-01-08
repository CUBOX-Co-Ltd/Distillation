from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from ultralytics import YOLO

from .distil_common import KnowledgeDistiler 
from .loss import KDLoss


class YOLO11Distiler(KnowledgeDistiler):
    def __init__(self,
                 fabric,
                 config,
                 trainset,
                 trainloader):
        super().__init__(fabric, config, trainset, trainloader)



    def init_model(self, ):
        self.teacher = YOLO().eval() 
        self.student = YOLO()
        self.discriminator = None

        if self.cfg.distillation_loss_type:
            self.distil_loss = KDLoss(self.cfg.temperature, self.cfg.alpha)
        else:
            pass

    def set_model_requires_grad(self, ):
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def init_optimizer(self,):
        # Optimizer class
        optimizer_class = None
        if 'offload' in self.cfg.pl_strategy:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer_class = DeepSpeedCPUAdam
        else:
            optimizer_class = optim.AdamW

        # Students
        student_parameters = list(filter(lambda p: p.requires_grad, self.student.parameters()))

        self.student_optimizer = optimizer_class(
            student_parameters,
            lr=self.cfg.student_lr,
            weight_decay=5e-4
        )
        
        if self.cfg.student_lr_scheduler is not None:
            student_scheduler_cls = getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.cfg.student_lr_scheduler,
                )
            self.student_lr_scheduler = student_scheduler_cls(self.student_optimizer, 
                gamma=0.9
            )

        # Discriminator
        if self.discriminator is not None:
            discriminator_parameters = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
            self.disc_optimizer = optimizer_class(
                discriminator_parameters,
                lr=self.cfg.disc_lr,
                weight_decay=5e-4
            )

            disc_scheduler_cls = getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.cfg.disc_lr_scheduler,
                )

            self.disc_scheduler = disc_scheduler_cls(
                self.disc_optimizer,
                gamma=0.9
            )

            self.disc_update_counter = 0

    def fabric_setup(self,):
        self.student, self.optimizer = self.fabric.setup(self.student, self.optimizer)
        try:
            from torchinfo import summary
            model_stats = summary(self.student, (1,1,1,1), verbose=0)
            self.fabric.print(str(model_stats))
        except Exception as e:
            print(e)


    def train_step(self, batch):
        images, targets = batch['img'], batch['labels']

        student_logits = student_model(images)

        with torch.no_grad():
            teacher_logits = teacher_model(images)


        loss = self.distil_los(student_logits, teacher_logits, targets)

        return loss

    def train(self,):
        for epoch in range(10):
            self.fabric.print(f"Epoch {epoch + 1}")
            self.student.train()
            total_loss = 0

            self.trainloader = tqdm(self.trainloader) if self.fabric.global_rank == 0 else self.trainloader
            for batch in self.trainloader:
                loss = self.train_step(batch)

                # Backpropagation
                self.optimizer.zero_grad()
                self.fabric.backward(loss, model=self.student)
                self.optimizer.step()

                total_loss += loss.item()

        fabric.print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader)}")

