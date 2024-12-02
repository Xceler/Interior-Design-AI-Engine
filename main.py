import os 
import logging 
from config.model_config import ModelConfiguration 
from utils.data_preprocessor import DataPreprocessor 
from utils.model_utils import ModelUtilities 
from models.classification_model import InteriorClassification
from models.feature_extraction_model import InteriorFeatureExtractionModel
from models.style_recommendation_model import StyleRecommendationModel 
from models.design_generation_model import DesignGenerationModel 

logging.basicConfig(level =logging.INFO) 


class InteriorDesignAI:
    def __init__(self):
        self.dataset_config = ModelConfiguration.get_dataset_config() 
        self.classification_config = ModelConfiguration.get_classification_config() 
        self.feature_extraction_config = ModelConfiguration.get_feature_extraction_config() 


        self.data_preprocessor = DataPreprocessor(
            dataset_path = self.dataset_config['base_path']
        )

        self.classification_model = InteriorClassification(
            input_shape = self.classification_config['input_shape'],
            num_classes = self.classification_config['num_classes']
        )

        self.feature_extraction_model = InteriorFeatureExtractionModel(
            pretrained_weights = self.feature_extraction_config['pretrained_weights']
        )

        self.style_recommendation_model = StyleRecommendationModel()
        self.design_generation_model = DesignGenerationModel()

    
    def preprocess_dataset(self):
        logging.info("Preprocessing dataset....")
        processed_data = self.data_preprocessor.prepare_dataset()
        logging.info("Dataset Preprocessing Completed")
        return processed_data 
    

    def train_models(self, processed_data):
        train_data, train_labels = self._prepare_training_data(processed_data)

        logging.info("Training Classification Model...")
        self.classification_model.train(train_data, train_labels)
        logging.info("Training Feature Extraction Model...")
        self.feature_extraction_model.train(train_data) 
        logging.info("Training Style Recommendation Model...")
        self.style_recommendation_model.train(train_data) 
        logging.info("Training Design Generation Model ...")
        self.design_generation_model.train(train_data) 

    
    def _prepare_training_data(self, processed_data):
        train_data = [entry['image_features'] for entry in processed_data]
        train_labels = [entry['style_label'] for entry in processed_data]



        return train_data, train_labels
    
    def recommend_designs(self, input_image_path):
        logging.info(f"Generating Recommendations for {input_image_path}...")
        input_features = self.data_preprocessor.extract_features(input_image_path) 
        style_classification = self.classification_model.classify(input_features) 
        recommendations = self.style_recommendation_model.recommend(
            input_features, 
            top_k = 5
        )
        logging.info(f"Recommendations Generated : {recommendations}")

    
    def generate_design(self, style_descriptions):
        logging.info(f"Generating design for description : {style_descriptions}")
        return self.design_generation_model.generate(style_descriptions) 


def main():
    try:
        interior_design_ai = InteriorDesignAI() 
        logging.info("Initialized Interior Design AI System.")

        logging.info("Starting Dataset Preprocessing ...")
        processed_data = interior_design_ai.preprocess_dataset() 

        logging.info("Training Models ...")
        interior_design_ai.train_models(processed_data) 

        sample_image_path = 'System/data/Data_set/train/Bathroom/bath_1_aug_0.jpg'
        recommendations = interior_design_ai.recommend_designs(sample_image_path) 
        logging.info(f"Design Recommendations For {sample_image_path} : {recommendations}")

        generated_design = interior_design_ai.generate_design("Modern Minimalist Bathroom")
        logging.infof("Generated Design : {generated_design}")
    
    except Exception as e:
        logging.error(f"An Error Occured : {e}")


    

if __name__ == "__main__":
    main()
