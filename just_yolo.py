from ultralytics import YOLO
import os
from distiller import get_yolo_model
import torch
from torchinfo import summary

# Load a model
model = YOLO("yolo11n.pt")

print('model.stride', model.stride)
print('model.taskmap', model.task_map)
print('model type', type(model.model))
print(type(model.model.criterion))

# model_stats = summary(model.eval(), (1,3,640,640), verbose=0)
# print(str(model_stats))

# print(model)

# model2 = get_yolo_model("/purestorage/project/tyk/3_CUProjects/Distillation/ultralytics/cfg/models/11/yolo11.yaml")
# # print('model.type', type(model2.model))
# # model_stats = summary(model2.eval(), (1,3,640,640), verbose=0)
# # print(str(model_stats))
# print(model2.model)
# print(type(model2.model.criterion))
# x = torch.randn(1, 3, 640, 640)

# # out = model.model._predict_once(x)

# # # for i, o in enumerate(out):
# # #     if isinstance(o, torch.Tensor):  # 텐서일 경우 shape 출력
# # #         print(f"Shape of output[{i}]: {o.shape}")
# # #     else:
# # #         print(f"output[{i}] is not a tensor, type: {type(o)}")


# out = model2.model._predict_once(x)

# for i, o in enumerate(out):
#     if isinstance(o, torch.Tensor):  # 텐서일 경우 shape 출력
#         print(f"Shape of output[{i}]: {o.shape}")
#     else:
#         print(f"output[{i}] is not a tensor, type: {type(o)}")

# # # Train the model
# # train_results = model.train(
# #     data="/purestorage/project/tyk/3_CUProjects/Distillation/ultralytics/cfg/datasets/coco.yaml",  # path to dataset YAML
# #     epochs=100,  # number of training epochs
# #     imgsz=640,  # training image size
# #     batch=128, 
# #     optimizer="SGD",
# #     device=os.getenv("CUDA_VISIBLE_DEVICES"), workers=12, plots=True
# # )