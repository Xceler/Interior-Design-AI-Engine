import numpy as np 
import tensorflow as tf 
from typing import Any, Dict, List 

class ModelUtilities:
    @staticmethod 
    def load_model(model_path: str) -> Any:
        try:
            return tf.keras.models.load_model(model_path) 
        except Exception as e:
            print(f"Error Loading Model : {e}")
            return None 
    

    @staticmethod
    def save_model(model: Any, save_path: str) -> bool:
        try:
            model.save(save_path) 
            return True 
        
        except Exception as e:
            print(f"Error Handling : {e}")
            return False 
        

    @staticmethod 
    def evaluate_model(model: Any, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        try:
            loss, accuracy = model.evaluate(test_data, test_labels) 
            return {
                'loss' : loss,
                'accuracy' : accuracy
            }
        
        except Exception as e:
            print(f"Error Evaluating Model :{e}")
            return {} 
    

    @staticmethod
    def predict(model: Any, input_data: np.ndarray) -> List[Any]:
        try:
            return model.predict(input_data) 
        except Exception as e:
            print(f"Error Making Predictions : {e}")
            return [] 
    