from distiller import get_yolo_model, get_yolo_models
import torch
import debugpy
debugpy.listen(('0.0.0.0', 5678))
debugpy.wait_for_client()
device = 'cuda'

teacher_version="/home/Distillation/ultralytics/cfg/models/11/yolo11s.yaml"
student_version="/home/Distillation/ultralytics/cfg/models/11/yolo11n.yaml"
task="detect"
# teacher, student = get_yolo_models(teacher_version, student_version)
from ultralytics.models.yolo.model import YOLO
teacher = YOLO(teacher_version, task=task).to(device)
student = YOLO(student_version, task=task).to(device)
teacher.to(device)
student.to(device)

teacher.model.args = teacher.args
student.model.args = student.args

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data.build import build_dataloader
data_yaml = '/home/Distillation/ultralytics/cfg/datasets/coco.yaml'
data = check_det_dataset(data_yaml) # yaml의 내용이 담긴 
trainset = YOLODataset(
    data["train"],
    data=data,
    task="detect",
    imgsz=640,
    augment=False,
    batch_size=1,)

trainloader = torch.utils.data.DataLoader(
                trainset, 
                batch_size=32, 
                shuffle=True, 
                num_workers=8,
                collate_fn=getattr(trainset, "collate_fn", None),)

for batch in trainloader:
    batch["img"] = (batch["img"].to(device, non_blocking=True).float() / 255)
    
    student_preds = student.model._predict_once(batch["img"])
    with torch.no_grad():
        teacher_preds = teacher.model._predict_once(batch["img"])

    print(student_preds[0].shape, student_preds[1].shape, student_preds[2].shape)
    print(teacher_preds[0].shape, teacher_preds[1].shape, teacher_preds[2].shape)
    break