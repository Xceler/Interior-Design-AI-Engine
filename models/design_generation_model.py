import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 

class DesignGenerationModel:
    def __init__(self):
        self.design_rule =  {
            "Modern Minimalist": {
                "color_palette": ["white", "gray", "black"],
                "furniture_guidelines": {
                    "sofa": ["low-profile", "clean-lines"],
                    "table": ["geometric", "minimal-decor"]
                }
            },
            "Bohemian": {
                "color_palette": ["terracotta", "mustard", "sage-green"],
                "furniture_guidelines": {
                    "sofa": ["textured", "plush"],
                    "table": ["wooden", "organic-shapes"]
                }
            },
            "Scandinavian": {
                "color_palette": ["white", "light-wood", "soft-blue"],
                "furniture_guidelines": {
                    "sofa": ["simple", "light-fabric"],
                    "table": ["natural-wood", "functional"]
                }
            },
            "Industrial Loft": {
                "color_palette": ["charcoal", "exposed-brick", "steel-gray"],
                "furniture_guidelines": {
                    "sofa": ["leather", "industrial"],
                    "table": ["metal", "reclaimed-wood"]
                }
            },
            "Mid-Century Modern": {
                "color_palette": ["teal", "orange", "wood-brown"],
                "furniture_guidelines": {
                    "sofa": ["angular", "retro-fabric"],
                    "table": ["round", "teak-wood"]
                }
            },
            "Japandi": {
                "color_palette": ["neutral-tones", "black", "wood"],
                "furniture_guidelines": {
                    "sofa": ["low-profile", "minimal"],
                    "table": ["simple", "light-wood"]
                }
            },
            "Coastal": {
                "color_palette": ["white", "blue", "sand"],
                "furniture_guidelines": {
                    "sofa": ["light-fabric", "casual"],
                    "table": ["weathered-wood", "nautical"]
                }
            },
            "Art Deco": {
                "color_palette": ["gold", "black", "rich-emerald"],
                "furniture_guidelines": {
                    "sofa": ["luxurious", "velvet"],
                    "table": ["glossy", "metal"]
                }
            },
            "Rustic": {
                "color_palette": ["earthy-brown", "green", "beige"],
                "furniture_guidelines": {
                    "sofa": ["plush", "wooden-frame"],
                    "table": ["rough-wood", "handcrafted"]
                }
            },
            "Contemporary": {
                "color_palette": ["neutral-tones", "black", "blue"],
                "furniture_guidelines": {
                    "sofa": ["bold-lines", "multi-color"],
                    "table": ["glass-top", "sleek"]
                }
            },
            "Traditional": {
                "color_palette": ["cream", "dark-wood", "burgundy"],
                "furniture_guidelines": {
                    "sofa": ["classic", "structured"],
                    "table": ["mahogany", "detailed-carvings"]
                }
            },
            "Transitional": {
                "color_palette": ["beige", "gray", "white"],
                "furniture_guidelines": {
                    "sofa": ["neutral", "blended-styles"],
                    "table": ["simple", "multi-material"]
                }
            },
            "Eclectic": {
                "color_palette": ["varied-bright", "earth-tones"],
                "furniture_guidelines": {
                    "sofa": ["mix-and-match", "colorful"],
                    "table": ["vintage", "quirky"]
                }
            },
            "Mediterranean": {
                "color_palette": ["blue", "white", "terracotta"],
                "furniture_guidelines": {
                    "sofa": ["woven", "comfortable"],
                    "table": ["stone", "rustic"]
                }
            },
            "Farmhouse": {
                "color_palette": ["white", "gray", "light-wood"],
                "furniture_guidelines": {
                    "sofa": ["simple", "comfortable"],
                    "table": ["wooden", "distressed"]
                }
            }
        }

        self.placement_model = KMeans(n_clusters = 3)
    
    def generate_design_layout(self, style, detected_objects):
        style_rules = self.design_rule(style, {}) 

        layout_suggestions = {
            "style" : style,
            "color_palette" : style_rules.get("color_palette", []),
            "furniture_placement" : {},
            "design_recommendations" : []
        }


        for obj_type, count in detected_objects.get("count", {}).items():
            guidelines = style_rules.get("furniture_guidelines", {}).get(obj_type, [])

            layout_suggestions['furniture_placement'][obj_type] = {
                "count" : count, 
                "recommended_style" : guidelines
            }
        
        layout_suggestions['design_recommendations'] = self._generate_recommendations(
            style, detected_objects
        )

        return layout_suggestions 
    
    def _generate_recommendations(self, style, detected_objects):
        recommendations = [] 
        if style == "Modern Minimalist":
            recommendations.append("Consider adding a sleek, low-profile sofa to match the minimal decor.")
            recommendations.append("Use geometric table designs to enhance the modern feel.")
        elif style == "Bohemian":
            recommendations.append("Layer textured fabrics and colorful throw pillows for a bohemian touch.")
            recommendations.append("Include rustic, wooden tables and chairs for an organic look.")
        elif style == "Scandinavian":
            recommendations.append("Opt for functional and simple furniture with light-wood finishes.")
            recommendations.append("Add soft, cozy blankets and light fabric sofa covers.")
        elif style == "Industrial Loft":
            recommendations.append("Incorporate metal elements and exposed brick features.")
            recommendations.append("Use leather sofas and industrial-style light fixtures.")
        elif style == "Mid-Century Modern":
            recommendations.append("Choose angular sofas with retro patterns.")
            recommendations.append("Add teak wood tables for a mid-century vibe.")
        elif style == "Japandi":
            recommendations.append("Select minimalistic furniture with light wood and neutral tones.")
            recommendations.append("Keep decoration simple, using natural materials like bamboo or linen.")
        elif style == "Coastal":
            recommendations.append("Use light, airy fabrics and incorporate nautical elements.")
            recommendations.append("Add weathered-wood furniture and whitewashed decor.")
        elif style == "Art Deco":
            recommendations.append("Incorporate bold geometric shapes and luxury materials.")
            recommendations.append("Use gold accents and mirrored surfaces to add elegance.")
        elif style == "Rustic":
            recommendations.append("Add earthy tones and handcrafted wooden furniture.")
            recommendations.append("Use textured fabric cushions and woven items for added warmth.")
        elif style == "Contemporary":
            recommendations.append("Include bold, clean lines and unique shapes.")
            recommendations.append("Use a mix of materials like glass, metal, and wood for contrast.")
        elif style == "Traditional":
            recommendations.append("Use rich, deep colors and classic furniture.")
            recommendations.append("Incorporate detailed woodwork and ornate patterns.")
        elif style == "Transitional":
            recommendations.append("Blend traditional and contemporary pieces for a balanced look.")
            recommendations.append("Use neutral tones and simple designs for a versatile decor.")
        elif style == "Eclectic":
            recommendations.append("Mix vintage and modern furniture for an unexpected look.")
            recommendations.append("Add unique, one-of-a-kind decorative pieces.")
        elif style == "Mediterranean":
            recommendations.append("Incorporate terra-cotta tiles and stone surfaces.")
            recommendations.append("Use woven materials and rustic wooden furniture.")
        elif style == "Farmhouse":
            recommendations.append("Add cozy, simple furniture with a weathered finish.")
            recommendations.append("Use whitewashed wood and soft, natural fabrics.")
        
        return recommendations


        


