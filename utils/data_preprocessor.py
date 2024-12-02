import os 
import cv2 
import numpy as np 
import pandas as pd 
from typing import Dict, List, Any 
from sklearn.cluster import KMeans 


  
class DataPreprocessor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    
    def preprocess_image(self, image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:

        image = cv2.imread(image_path) 
        if image is None: 
            raise FileNotFoundError(f"Image Not Found At Path : {image_path}")
        
        image = cv2.resize(image, target_size) 
        image = image.astype("float32")  / 255.0 

        return image 
    

    def extract_features(self, image_path: str) -> Dict[str, Any]:
        processed_image = self.preprocess_image(image_path) 


        features = {
            "dimensions" : processed_image.shape,
            "color_distribution" : self._get_color_distribution(processed_image),
            'dominant_colors' : self._extract_dominant_colors(processed_image)
        }

        return features 
    
    def _get_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]

        return {
            "red_mean" : np.mean(r_channel),
            "green_mean" : np.mean(g_channel),
            "blue_mean"  : np.mean(b_channel)

        }    
    
    def _extract_dominant_colors(self, image : np.ndarray, num_colors : int = 5) -> List[tuple]:
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters = num_colors, random_state = 42) 
        kmeans.fit(pixels) 

        dominant_colors = kmeans.cluster_centers_ 
        return [tuple(color) for color in dominant_colors]
    
    def prepare_dataset(self) -> List[Dict[str, Any]]:
        processed_data = [] 
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(self.dataset_dir, split) 
            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category) 

                if not os.path.isdir(category_path):
                    continue 


                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name) 
                    if not image_name.lower().endswith(("png", "jpg", "jpeg")):
                        continue 

                    try:
                        image_features = self.extract_features(image_path) 
                        processed_data.append({
                            "split" : split,
                            "category" : category, 
                            "image_filename" : image_name,
                            **image_features
                        })
                    
                    except Exception as e:
                        print(f"Error Processing {image_path} : {e}")
        return processed_data 
    

    def save_processed_data(self, output_path: str):
        processed_data = self.prepare_dataset() 
        df = pd.DataFrame(processed_data) 
        df.to_csv(output_path, index= False) 



def run_processor(dataset_dir: str, output_csv: str):
    preprocessor = DataPreprocessor(dataset_dir = dataset_dir)
    preprocessor.save_preprocessed_data(output_csv) 
                
