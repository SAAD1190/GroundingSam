import supervision as sv
import os
import torch
from typing import List
import cv2
from tqdm.notebook import tqdm
from groundingdino.util.inference import Model

GROUNDING_DINO_CONFIG_PATH = os.path.join(".", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(".", "weights", "groundingdino_swint_ogc.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert(os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
assert(os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

def enhance_class_name(class_names: List[str]) -> List[str]:
  return [
      f"all {class_name}s"
      for class_name
      in class_names
  ]

class Annotator:
  def __init__(self, classes, images_dir = "./data/", annotations_dir = "./annotations/", images_extensions = ['jpg', 'jpeg', 'png']):
    self.classes = classes
    self.images_dir = images_dir
    self.annotations_dir = annotations_dir
    self.images_extensions = images_extensions
    
    self.image_paths = sv.list_files_with_extensions(
      directory=self.images_dir,
      extensions=self.images_extensions)
    
    self.detections = None
    self.images = {}
    self.annotations = {}

  def get_boxes(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
    for image_path in tqdm(self.image_paths):
      image_name = image_path.name
      image_path = str(image_path)
      image = cv2.imread(image_path)

      self.detections = grounding_dino_model.predict_with_classes(
          image=image,
          classes=enhance_class_name(class_names=self.classes),
          box_threshold=BOX_TRESHOLD,
          text_threshold=TEXT_TRESHOLD
      )
      # We made the detections var  an attribute  
      self.detections = self.detections[self.detections.class_id != None] 

      self.images[image_name] = image
      self.annotations[image_name] = self.detections
  
  def get_box(self, image_path, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
    image = cv2.imread(image_path)

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=self.classes),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    # We made the detections var  an attribute  
    detections = detections[detections.class_id != None]
    return detections 
  
  def to_pascal_voc(self, MIN_IMAGE_AREA_PERCENTAGE=0.002, MAX_IMAGE_AREA_PERCENTAGE=0.80, APPROXIMATION_PERCENTAGE=0.75):
    sv.Dataset(
        classes=self.classes,
        images=self.images,
        annotations=self.annotations
    ).as_pascal_voc(
        annotations_directory_path=self.annotations_dir,
        min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
        max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
        approximation_percentage=APPROXIMATION_PERCENTAGE
    )
  
  def plot_boxes(self):
    plot_images = []
    plot_titles = []

    box_annotator = sv.BoxAnnotator()

    for image_name, self.detections in self.annotations.items():
      image = self.images[image_name]
      plot_images.append(image)
      plot_titles.append(image_name)

      labels = [
          f"{self.classes[class_id]} {confidence:0.2f}"
          for _, _, confidence, class_id, _
          in self.detections]
      annotated_image = box_annotator.annotate(scene=image.copy(), detections=self.detections, labels=labels)
      plot_images.append(annotated_image)
      title = " ".join(set([
          self.classes[class_id]
          for class_id
          in self.detections.class_id
      ]))
      plot_titles.append(title)

    sv.plot_images_grid(
        images=plot_images,
        titles=plot_titles,
        grid_size=(len(self.annotations), 2),
        size=(2 * 4, len(self.annotations) * 4)
    )