import torch 
from transformers import CLIPProcessor, CLIPModel 
import numpy as np 


class StyleRecommendationModel:
    def __init__(self, model_name = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name) 
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.design_styles = [
            "Modern Minimalist",
            "Scandinavian",
            "Industrial Loft",
            "Bohemian",
            "Mid-Century Modern",
            "Japandi",
            "Coastal",
            "Art Deco",
            "Rustic",
            "Contemporary",
            "Traditional",
            "Transitional",
            "Eclectic",
            "Mediterranean",
            "Farmhouse"
        ]

    
    def _preprocess_image(self, image_path):
        image =self.processor(
            text = None,
            images = image_path,
            return_tensors = 'pt'
        )['pixel_values'].squeeze()

        return image 
    
    def recommend_styles(self, detected_objects, top_k = 3):

        object_description = ', '.join(
            [f"{count} {obj}" for obj, count in detected_objects.get('count', {}).items()]
        )

        image_inputs = self.processor(
            text = self.design_styles,
            images = object_description,
            return_tensors = 'pt',
            padding = True
        )


        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**image_inputs)
            text_embeddings = self.model.get_text_features(**image_inputs)

        
        similarity = torch.nn.functional.cosine_similarity(
            image_embeddings, 
            text_embeddings
        )


        top_styles_indices = similarity.topk(top_k).indices
        recommended_styles = [self.design_styles[idx] for idx in top_styles_indices]

        return recommended_styles 
    
    def explain_style_recommendation(self, recommended_styles, detected_objects):
        style_explanations = {}

        for style in recommended_styles:
            explanation = f"Based ON the {style} style , we recommend:"


            style_recommendations =  {
                "Modern Minimalist": {
                    "furniture": ["clean-lined sofa", "geometric coffee table"],
                    "colors": ["neutral tones", "white", "gray"]
                },
                "Scandinavian": {
                    "furniture": ["light wood dining table", "modern armchair"],
                    "colors": ["white", "soft pastels", "natural wood"]
                },
                "Industrial Loft": {
                    "furniture": ["metal coffee table", "exposed shelving"],
                    "colors": ["gray", "black", "warm brown"]
                },
                "Bohemian": {
                    "furniture": ["textured armchair", "wooden side tables"],
                    "colors": ["warm earth tones", "rich patterns"]
                },
                "Mid-Century Modern": {
                    "furniture": ["teak dining chairs", "low-profile sofa"],
                    "colors": ["teal", "mustard yellow", "walnut brown"]
                },
                "Japandi": {
                    "furniture": ["simple wooden table", "low bed frame"],
                    "colors": ["white", "soft beige", "light wood"]
                },
                "Coastal": {
                    "furniture": ["white wicker chairs", "light wood table"],
                    "colors": ["blue", "white", "sand tones"]
                },
                "Art Deco": {
                    "furniture": ["glossy side table", "geometric chair"],
                    "colors": ["gold", "black", "emerald green"]
                },
                "Rustic": {
                    "furniture": ["wooden bench", "cozy armchair"],
                    "colors": ["earthy brown", "cream", "deep green"]
                },
                "Contemporary": {
                    "furniture": ["sleek sofa", "modern dining set"],
                    "colors": ["neutral shades", "black", "white"]
                },
                "Traditional": {
                    "furniture": ["carved wooden cabinet", "classic armchair"],
                    "colors": ["deep red", "navy blue", "gold"]
                },
                "Transitional": {
                    "furniture": ["mix of modern and traditional pieces"],
                    "colors": ["gray", "beige", "muted blues"]
                },
                "Eclectic": {
                    "furniture": ["mismatched chairs", "bold-patterned sofa"],
                    "colors": ["vibrant colors", "contrasting patterns"]
                },
                "Mediterranean": {
                    "furniture": ["terracotta planter", "woven basket chair"],
                    "colors": ["white", "blue", "sunset orange"]
                },
                "Farmhouse": {
                    "furniture": ["wooden dining table", "simple armchair"],
                    "colors": ["white", "gray", "light wood"]
                }
            }

            if style in style_recommendations:
                rec = style_recommendations[style]
                explanation += f"{rec['furniture'][0]} with {rec['çolors'][0]} palette."

            
            style_explanations[style] =explanation 

        
        return style_explanations
