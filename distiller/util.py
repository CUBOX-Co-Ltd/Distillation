from ultralytics.models.yolo.model import YOLO
def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d
    
# For detection dataset
def preprocess_batch(batch, device, precision):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch["img"] = (batch["img"].to(device, non_blocking=True).float() / 255).to(precision)
    return batch

def get_yolo_model(model_version):
    model = YOLO(model_version)
    return model

def get_yolo_models(
    teacher_version="yolo11n.pt",
    student_version="yolo11n.pt"
    ):
    """
        YOLO 클래스 = ultralytics.models.yolo.model.YOLO
    """
    
    teacher = YOLO(teacher_version).eval()
    student = YOLO(student_version)

    return teacher, student