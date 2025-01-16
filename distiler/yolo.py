from ultralytics.models.yolo.model import YOLO
from .registry import register_method

_target_class = YOLO

@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    feat.append(x)
    x = self.layer2(x)
    feat.append(x)
    x = self.layer3(x)
    feat.append(x)
    x = self.layer4(x)
    feat.append(x)

    return (x, feat) if requires_feat else x


@register_method
def backbone_cbs_info(self, stage):
    """
    SPPF feature shape = Stage 4 feature shape 
    """
    if self.model_type == 'yolo11n':
        if stage == 0:
            index = 0
            shape = (16, 320, 320)
        elif stage == 1:
            index = 1
            shape = (32, 160, 160)   
        elif stage == 2:
            index = 2
            shape = (64, 80, 80)
        elif stage == 3:
            index = 3
            shape = (128, 40, 40)
        elif stage == 4:
            index = 4
            shape = (256, 20, 20)
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.model_type == 'yolo11x':
        if stage == 0:
            index = 0
            shape = (96, 320, 320)
        elif stage == 1:
            index = 1
            shape = (192, 160, 160)
        elif stage == 2:
            index = 2
            shape = (384, 80, 80)
        elif stage == 3:
            index = 3
            shape = (768, 40, 40)
        elif stage == 4:
            index = 4
            shape = (768, 20, 20)
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.model_type}')
    return index, shape

@register_method
def neck_cbs_info(self, stage):
    if self.model_type == 'yolo11n':
        if stage == 0:
            index = 0
            shape = (64, 40, 40)
        elif stage == 1:
            index = 1
            shape = (128, 20, 20)  
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.model_type == 'yolo11x':
        if stage == 0:
            index = 0
            shape = (384, 40, 40)
        elif stage == 1:
            index = 1
            shape = (786, 20, 20)
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.model_type}')
    return index, shape


@register_method
def head_info(self, stage, stream_type):
    if self.model_type == 'yolo11n':
        if stage == 0 and stream_type == 'long':
            index = 0
            shape = (80, 80, 80)
        elif stage == 0 and stream_type == 'short':
            index = 0
            shape = (64, 80, 80)
        elif stage == 1 and stream_type == 'long':
            index = 0
            shape = (80, 40, 40)
        elif stage == 1 and stream_type == 'short':
            index = 0
            shape = (64, 40, 40)
        elif stage == 2 and stream_type == 'long':
            index = 0
            shape = (80, 20, 20)
        elif stage == 2 and stream_type == 'short':
            index = 0
            shape = (64, 20, 20)
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-2)')
    elif self.model_type == 'yolo11x':
        if stage == 0 and stream_type == 'long':
            index = 0
            shape = (80, 80, 80)
        elif stage == 0 and stream_type == 'short':
            index = 0
            shape = (64, 80, 80)
        elif stage == 1 and stream_type == 'long':
            index = 0
            shape = (80, 40, 40)
        elif stage == 1 and stream_type == 'short':
            index = 0
            shape = (64, 40, 40)
        elif stage == 2 and stream_type == 'long':
            index = 0
            shape = (80, 20, 20)
        elif stage == 2 and stream_type == 'short':
            index = 0
            shape = (64, 20, 20)
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-2)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.model_type}')
    return index, shape