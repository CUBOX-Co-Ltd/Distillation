import argparse
import yaml

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
import wandb
import os

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data.build import build_dataloader

from configs import Config

def yaml_load(file_path: str):
    f = open(file_path, 'r')
    data = yaml.safe_load(f)
    f.close()

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--mode', type=str, default='yolo11')
    parser.add_argument('--config_path', type=str, default='configs/yolo_basic_distil.yaml')

    args = parser.parse_args()

    config = Config(**yaml_load(args.config_path))


    wandb.login(key=config.wandb_key, host=config.wandb_host)
    wandb_logger = WandbLogger(
        project=config.wandb_project_name, 
        name=config.wandb_run_name + '-' + os.environ.get('SLURM_JOBID', ''), 
        config=config)
    

    fabric = Fabric(
        accelerator='cuda', 
        num_nodes=args.nnodes,
        devices=args.ngpus, 
        strategy=config.pl_strategy,
        precision=config.pl_precision,
        loggers=wandb_logger,
    )

    fabric.launch()


    from torchinfo import summary
    from ultralytics import YOLO

    # YOLO11n 및 YOLO11x 모델 로드
    model_n = YOLO("yolo11n.pt")
    model_x = YOLO("yolo11x.pt")

    # 요약 정보 출력
    print("YOLO11n Summary:")
    summary(model_n.model, input_size=(1, 3, 640, 640), verbose=1)

    print("YOLO11x Summary:")
    summary(model_x.model, input_size=(1, 3, 640, 640), verbose=1)

    # YOLO11n ONNX 변환
    onnx_path_n = model_n.export(format="onnx")
    print(f"YOLO11n ONNX 모델 저장 경로: {onnx_path_n}")

    # YOLO11x ONNX 변환
    onnx_path_x = model_x.export(format="onnx")
    print(f"YOLO11x ONNX 모델 저장 경로: {onnx_path_x}")

    # if config.trainset == 'smartphone':
    #     from ultralytics.data.dataset import YOLODataset
    #     data_yaml = '/purestorage/project/tyk/3_CUProjects/Distillation/data/smartphone/smartphone.yaml'
    #     data = check_det_dataset(data_yaml) # yaml의 내용이 담긴 dict
    #     trainset = YOLODataset(
    #         data["train"],
    #         data=data,
    #         task="detect",
    #         imgsz=640,
    #         augment=False,
    #         batch_size=1,)

    #     # build_dataloader를 쓰면 ultralytics.data.build.InfiniteDataLoader가 나온다
    #     # trainloader = build_dataloader(trainset, batch=16, workers=4, shuffle=True)

    # if args.mode == 'yolo11':
    #     from distiler import YOLO11Distiler
    #     distiler = YOLO11Distiler(fabric=fabric, config=config, trainset=trainset)
    #     distiler.train()

    # if args.mode == 'test':
    #     ...


