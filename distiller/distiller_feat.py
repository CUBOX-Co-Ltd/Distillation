import torch
import torch.nn as nn

from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from ultralytics import YOLO

import importlib
from tqdm import tqdm

from .distil_common import BaseDistiller
from .blocks import init_weights
from .projector import projectors
from .util import preprocess_batch
# from .loss import FeatureDistillationLoss

class Student_with_Proejctor(nn.Module):
    def __init__(self, config, teacher, student):
        self.config = config
        self.student = student
        # Teacher, Student
        model_setup_func_map = {
            ("YOLO", "YOLO"): self.setup_yolo_with_yolo,
            ("GDINO", "YOLO"): self.setup_yolo_setup_yolo_with_gdino
        }
        key = (teacher.__class__.__name__, student.__class__.__name__)
        if key in function_map:
            model_setup_func_map[key](teacher, student)
        else:
            print(f"No function defined for {key}")

    def setup_yolo_with_yolo(self, teacher, student):
        self.projector_backbone = nn.ModuleDict()
        self.projector_neck = nn.ModuleDict()
        self.projector_deep_head = nn.ModuleDict()
        self.projector_mid_head = nn.ModuleDict()
        self.projector_low_head = nn.ModuleDict()

        # backbone + sppf
        for stage in self.config.backbone_cbs:
            _, student_shape = student.backbone_cbs_info(stage)
            student_channels, _, _ = student_shape

            _, teacher_shape = teacher.backbone_cbs_info(stage)
            teacher_channels, _, _ = teacher_shape


            projector = projectors[self.config.projector_type](
                student_channels, max(student_channels, teacher_channels) 
            )

            self.projector_backbone[str(stage)] = projector

        self.projector_backbone.apply(init_weights)

        # neck_cbs: [1,2]
        for stage in self.config.neck_cbs:
            _, student_shape = student.neck_cbs_info(stage)
            student_channels, _, _ = student_shape

            _, teacher_shape = teacher.neck_cbs_info(stage)
            teacher_channels, _, _ = teacher_shape


            projector = projectors[self.config.projector_type](
                student_channels, max(student_channels, teacher_channels) 
            )

            self.projector_neck[str(stage)] = projector
        
        self.projector_neck.apply(init_weights)

        # head_deep_cbs: [1, 2]
        for stage in self.config.head_deep_cbs:
            _, student_shape = student.neck_cbs_info(stage)
            student_channels, _, _ = student_shape

            _, teacher_shape = teacher.neck_cbs_info(stage)
            teacher_channels, _, _ = teacher_shape


            projector = projectors[self.config.projector_type](
                student_channels, max(student_channels, teacher_channels) 
            )

            self.projector_deep_head[str(stage)] = projector

        self.projector_deep_head.apply(init_weights)
        
        for stage in self.config.head_mid_cbs:
            _, student_shape = student.neck_cbs_info(stage)
            student_channels, _, _ = student_shape

            _, teacher_shape = teacher.neck_cbs_info(stage)
            teacher_channels, _, _ = teacher_shape


            projector = projectors[self.config.projector_type](
                student_channels, max(student_channels, teacher_channels) 
            )

            self.projector_mid_head[str(stage)] = projector

        self.projector_mid_head.apply(init_weights)


        for stage in self.config.head_low_cbs:
            _, student_shape = student.neck_cbs_info(stage)
            student_channels, _, _ = student_shape

            _, teacher_shape = teacher.neck_cbs_info(stage)
            teacher_channels, _, _ = teacher_shape


            projector = projectors[self.config.projector_type](
                student_channels, max(student_channels, teacher_channels) 
            )

            self.projector_low_head[str(stage)] = projector

        self.projector_low_head.apply(init_weights)


    def setup_yolo_with_gdino(self, teacher, student):
        pass


    def forward(self, batch):
        pass
        # logits_student, feat_student = self.student.model.forward_features(batch["img"], True)
        
        # for stage in self.config.backbone_cbs:
            

class CWDLoss(nn.Module):
    def __init__(self, tau):
        self.tau = tau
    
    def forward(self, feats_s, feats_t):
        assert len(feat_s) == len(feat_t)
        losses = []

        for idx, feat_s, feat_t in enumerate(zip(feats_s, feats_t)):
            assert feat_s == feat_t
            N, C, H, W = feat_s.shape

            softmax_pred_t = F.softmax(feat.view(-1, H*W) / self.tau, dim=1) # 픽셀별로 확률값으로 변환

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))

        return sum(losses)



class FeatuerDistiller(BaseDistiller):
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
        self.teacher = teacher
        self.student = Student_with_Proejctor(teacher, student)

    def set_model_requires_grad(self, ):
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        if self.cfg.train_all_params:
            for param in self.student.parameters():
                param.requires_grad = True
        else:
            pass


    def init_optimizer(self, ):
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

        student_output = self.student(batch)

        student_feats = self.student.model.extracted_features
        with torch.no_grad():
            self.teacher.eval()
            teacher_output = self.teacher.model._predict_once(batch["img"])

            teacher_feats = self.teacher.model.extracted_features
        # loss = 

    def train(self,):
        for epoch in range(self.config.num_epochs):
            self.fabric.print(f"Epoch {epoch + 1}")
            total_loss = 0

            self.trainloader = tqdm(self.trainloader) if self.fabric.global_rank == 0 else self.trainloader
            for batch in self.trainloader:
                loss = self.train_step(batch)

                # Backpropagation
                self.optimizer.zero_grad()
                self.fabric.backward(loss, model=self.student)
                self.optimizer.step()
