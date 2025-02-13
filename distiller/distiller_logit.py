import torch

from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from ultralytics import YOLO

import importlib
from tqdm import tqdm

from .distil_common import BaseDistiller
from .util import preprocess_batch
from .loss import BasicDistillationLoss, LogitDistillationLoss


class LogitDistiler(BaseDistiller):
    def __init__(self,
                 fabric,
                 config,
                 trainset,
                 teacher,
                 student,
                 trainloader=None):
        super().__init__(fabric, config, trainset, teacher, student, trainloader)



    def init_model(self, teacher, student):
        self.teacher = teacher
        self.student = student

        print('t args', self.teacher.model.args)
        print('s args', self.student.model.args)
        # print('loss type', type(self.teacher.model.criterion))
        # 파라미터 확인
        # print("Parameters in YOLO object:", list(self.student.parameters()))
        # print("Parameters in YOLO.model:", list(self.student.model.parameters()))
        # for name, param in self.student.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        self.discriminator = None

        if self.cfg.distillation_loss_type == 'basic':
            self.distil_loss = BasicDistillationLoss(self.cfg.temperature, self.cfg.alpha)
        elif self.cfg.distillation_loss_type == 'logit':
            self.response_criterion = LogitDistillationLoss(self.cfg.temperature)
        else:
            pass

    def set_model_requires_grad(self, ):
        self.teacher.requires_grad_(False)
        # self.teacher.eval()

        if self.cfg.train_all_params:
            for param in self.student.parameters():
                param.requires_grad = True
        else:
            pass

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
        self.student, self.optimizer = self.fabric.setup(self.student, self.student_optimizer)
        self.teacher = self.fabric.setup(self.teacher)
        try:
            from torchinfo import summary
            model_stats = summary(self.student, (1,1,1,1), verbose=0)
            self.fabric.print(str(model_stats))
        except Exception as e:
            print(e)


    def train_step(self, batch):
        """
        Args:
            batch : dict
                'im_file', 
                'ori_shape', 
                'resized_shape', 
                'ratio_pad', 
                'img', 
                'cls', 
                'bboxes', 
                'batch_idx'

        Model output:
            student_output :'tuple'
                Element 0: Tensor with shape torch.Size([16, 84, 8400])
                Element 1: List with length 3
                List Item 0: Tensor with shape torch.Size([16, 144, 80, 80])
                List Item 1: Tensor with shape torch.Size([16, 144, 40, 40])
                List Item 2: Tensor with shape torch.Size([16, 144, 20, 20])
        """

        batch = preprocess_batch(batch, self.device, self.weight_dtype)
        
        # Forward pass
        # student_preds = self.student.model(batch['img'])
        # with torch.no_grad():
        #     # self.teacher.model.eval()
        #     teacher_preds = self.teacher.model(batch['img'])


        student_preds = self.student.model._predict_once(batch["img"])
        with torch.no_grad():
            teacher_preds = self.teacher.model._predict_once(batch["img"])

        print(student_preds[0].shape, student_preds[1].shape, student_preds[2].shape)
        print(teacher_preds[0].shape, teacher_preds[1].shape, teacher_preds[2].shape)
        # Loss
        response_loss = self.response_criterion(teacher_preds, student_preds)
        gt_loss = self.student.model.criterion(student_preds, batch)

        total_loss = self.cfg.gt_coeff * gt_loss + self.cfg.response_coeff * self.response_loss
        return total_loss

    def train(self,):
        for epoch in range(self.cfg.num_epochs):
            self.fabric.print(f"Epoch {epoch + 1}")
            total_loss = 0

            self.trainloader = tqdm(self.trainloader) if self.fabric.global_rank == 0 else self.trainloader
            for batch in self.trainloader:
                loss = self.train_step(batch)

                # Backpropagation
                self.optimizer.zero_grad()
                self.fabric.backward(loss, model=self.student)
                self.optimizer.step()

                total_loss += loss.item()

        self.fabric.print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader)}")

