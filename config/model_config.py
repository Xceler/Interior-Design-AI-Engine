from typing import Dict, Any 

class ModelConfiguration:
    CLASSIFICATION_MODEL_CONFIG: Dict[str, Any] = {
        "input_shape" : (224, 224, 3),
        'num_classes' : 10,
        'learning_rate' : 0.001,
        'dropout_rate' : 0.5, 
        'architecture' : 'ResNet50'
    }

    FEATURE_EXTRACTION_CONFIG : Dict[str, Any] = {
        'embedding_dim' : 512, 
        'architecture' : 'VGG16',
        'pretrained_weights' : 'imagenet'
    }

    STYLE_RECOMMENDATION_CONFIG: Dict[str, Any] = {
        'similarity_metric' : 'consine',
        'top_k_recommendations' : 5,
        'embedding_threshold' : 0.7
    }

    DESIGN_GENERATION_CONFIG: Dict[str, Any] = {
        'latent_dim' : 100,
        'generator_layers' : 4, 
        'discriminator_layers' : 3,
        'noise_dim' : 50
    }


    DATASET_CONFIG : Dict[str, Any] = {
        'base_path' : './System/data/Data_set',
        'train_folder' : 'train',
        'test_folder' : 'test',
        'val_folder' : 'val',
        'meta_data' : 'dataset_metadata.json',
        'train_split' : '0.8',
        'test_split' : '0.1',
        'val_split' : '0.1'
    }


    @classmethod 
    def get_classification_config(cls) -> Dict[str, Any]:
        return cls.CLASSIFICATION_MODEL_CONFIG 
    
    @classmethod 
    def get_feature_extraction_config(cls) -> Dict[str, Any]:
        return cls.FEATURE_EXTRACTION_CONFIG 
    
    @classmethod 
    def get_style_recommendation_config(cls) -> Dict[str, Any]:
        return cls.STYLE_RECOMMENDATION_CONFIG 
    
    @classmethod 
    def get_design_generation_config(cls) -> Dict[str, Any]:
        return cls.DESIGN_GENERATION_CONFIG
    
    @classmethod 
    def get_dataset_config(cls) -> Dict[str, Any]:
        return cls.DATASET_CONFIG 