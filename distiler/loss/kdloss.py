import torch
import torch.nn.functional as F
from torch import nn

class KDLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        # Teacher logits -> softmax with temperature
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss (KL Divergence)
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)

        # Standard classification loss (CrossEntropy)
        classification_loss = F.cross_entropy(student_logits, targets)

        # Weighted sum of losses
        return self.alpha * classification_loss + (1 - self.alpha) * distillation_loss