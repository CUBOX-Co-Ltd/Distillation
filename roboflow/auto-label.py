from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv
from pathlib import Path
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image


ontology=CaptionOntology({
    "milk bottle": "bottle",
    "blue cap": "cap"
})


IMAGE_DIR_PATH = f"/purestorage/project/tyk/3_CUProjects/Distillation/data/milk/images"
DATASET_DIR_PATH = f"/purestorage/project/tyk/3_CUProjects/Distillation/data/milk/dataset"

ANNOTATIONS_DIRECTORY_PATH = f"{DATASET_DIR_PATH}/train/labels"
IMAGES_DIRECTORY_PATH = f"{DATASET_DIR_PATH}/train/images"
DATA_YAML_PATH = f"{DATASET_DIR_PATH}/data.yaml"

# # Do autolableling
# base_model = GroundedSAM(ontology=ontology)
# dataset = base_model.label(
#     input_folder=IMAGE_DIR_PATH,
#     extension=".png",
#     output_folder=DATASET_DIR_PATH)

# Check dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

print(len(dataset))


mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

images = []
image_names = []
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)


for i, (image_path, image, annotation) in enumerate(dataset):
    if i == SAMPLE_SIZE:
        break
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=annotation)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=annotation)

    image_names.append(Path(image_path).name)
    images.append(annotated_image)



SAVE_PATH = "/purestorage/project/tyk/3_CUProjects/Distillation/data/milk/image_grid.png"

import cv2
# Target size for resizing all images
TARGET_SIZE = (512, 512)  # (width, height)

# Resize all images to the target size
resized_images = [cv2.resize(img, TARGET_SIZE) for img in images]

# Convert images to PyTorch tensors and normalize to [0, 1]
tensor_images = [T.ToTensor()(img) for img in resized_images]

# Create a grid from the images
grid = make_grid(tensor_images, nrow=4, padding=2, normalize=True)

# Convert the grid back to a PIL image
grid_image = T.ToPILImage()(grid)

# Save the grid to a file
grid_image.save(SAVE_PATH)
print(f"Image grid saved to {SAVE_PATH}")

# sv.plot_images_grid(
#     images=images,
#     titles=image_names,
#     grid_size=SAMPLE_GRID_SIZE,
#     size=SAMPLE_PLOT_SIZE)