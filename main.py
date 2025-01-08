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

    if config.trainset == 'smartphone':
        from ultralytics.data.dataset import YOLODataset
        data_yaml = '/purestorage/project/tyk/3_CUProjects/Distillation/data/smartphone/smartphone.yaml'
        data = check_det_dataset(data_yaml)

        print(data)
        print(type(data))
        trainset = YOLODataset(
            data["train"],
            data=data_yaml,
            task="detect",
            imgsz=640,
            augment=False,
            batch_size=1,)


        dataloader = build_dataloader(trainset, batch=16, workers=4, shuffle=True)

    if args.mode == 'yolo11':
        from distiler import YOLO11Distiler
        distiler = YOLO11Distiler(fabric=fabric, config=config, trainset=trainset, trainloader=trainloader)
        distiler.train()

    if args.mode == 'test':
        ...


