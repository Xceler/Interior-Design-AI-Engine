import torch 
from ultralytics import YOLO 
import cv2 
import numpy as np 

class InteriorFeatureExtractionModel:
    def __init__(self, pretrained_weights="/content/drive/MyDrive/system/yolo8n.pt"):
        try:
            # Load the model with the specified weights
            self.model = YOLO(pretrained_weights)
            print(f"Model loaded successfully with weights from {pretrained_weights}")
        except FileNotFoundError:
            print(f"Error: The file {pretrained_weights} does not exist. Please check the path.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise

        # Define interior classes for object detection
        self.interior_classes = [
            'kitchen', 'dining', 'living', 'bedroom', 'bathroom',
            'sofa', 'chair', 'table', 'bed', 'tv',
            'lamp', 'bookshelf', 'window', 'door'
        ]

    def train(self, dataset_path, epochs=50, imgsz=640):
        try:
            results = self.model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=imgsz
            )
            return results
        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise

    def detect_objects(self, image_path):
        try:
            results = self.model(image_path)[0]
            detected_objects = {
                "objects": [],
                "count": {}
            }

            for result in results.boxes:
                cls = int(result.cls[0])
                conf = float(result.conf[0])
                class_name = self.model.names[cls]

                x1, y1, x2, y2 = result.xyxy[0]
                object_info = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }

                detected_objects['objects'].append(object_info)
                detected_objects['count'][class_name] = detected_objects['count'].get(class_name, 0) + 1

            return detected_objects
        except Exception as e:
            print(f"An error occurred during object detection: {e}")
            raise

    def detect_features(self, detected_objects):
        try:
            features = {
                'object_composition': {},
                "spatial_analysis": {}
            }

            for obj in detected_objects['objects']:
                cls = obj['class']
                features['object_composition'][cls] = features['object_composition'].get(cls, 0) + 1

            features['spatial_analysis'] = {
                'object_density': len(detected_objects['objects']) / (640 * 640),
                'unique_object_types': len(set(obj['class'] for obj in detected_objects['objects']))
            }

            return features
        except Exception as e:
            print(f"An error occurred during feature detection: {e}")
            raise
