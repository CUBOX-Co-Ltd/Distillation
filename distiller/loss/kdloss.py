import torch
import torch.nn.functional as F
from torch import nn

class LogitDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, teacher_logits, student_logits):
        print(len(student_logits), len(teacher_logits)) # 3, 2
        # print('teacher_logits_1', teacher_logits[0].shape)

        student_logits = student_logits[0] # [16, 84, 8400]
        teacher_logits = teacher_logits[0] # [16, 144, 80, 80])

        print('shape', student_logits.shape, teacher_logits.shape)
        print('teacher_logits', teacher_logits[1].shape) # ([84, 8400])
        print('teacher_logits', teacher_logits[2].shape) # 
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)

class BasicDistillationLoss(nn.Module):
    """
    Logit distillation + Feature distillation
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def kl_divergence_loss(self, student_logits, teacher_logits):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)

    def feature_distillation_loss(self, student_features, teacher_features):
        loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            loss += F.mse_loss(sf, tf)
        return loss

    def forward(self, student_output, teacher_output):
        # Logit distillation (output[0])
        logit_loss = self.kl_divergence_loss(student_output[0], teacher_output[0])

        # Feature distillation (output[1])
        feature_loss = self.feature_distillation_loss(student_output[1], teacher_output[1])

        # Combined loss
        total_loss = self.alpha * logit_loss + (1 - self.alpha) * feature_loss
        return total_loss