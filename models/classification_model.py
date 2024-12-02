import tensorflow as tf 
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model 

class InteriorClassification:
    def __init__(self, input_shape = (224, 224, 3), num_classes = 5):
        self.input_shape = input_shape 
        self.num_classes = num_classes 
        self.model = self._build_model() 
    
    def _build_model(self):
        base_model = ResNet50(
            weights = '/content/drive/MyDrive/system/ResNet50_weights_tf_dim_ordering_tf_kernels.h5',
            include_top = False,
            input_shape = self.input_shape
        )

        for layer in base_model.layers:
            layer.trainable = False 

        
        x = base_model.output 
        x = GlobalAveragePooling2D()(x) 
        x = Dense(1024, activation = "relu")(x) 
        x = Dense(512, activation = "relu")(x) 

        predictions = Dense(
            self.num_classes,
            activation = 'softmax',
            name= "Interior_Classification"
        )(x) 

        model = Model(inputs = base_model.input, outputs = predictions) 

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate =0.00001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model 
    

    def train(self, train_data, validation_data, epochs = 10):
        return self.model.fit(
            train_data,
            validation_data = validation_data,
            epochs = epochs
        )
    
    def predict(self, image):
        preprocessed_image = preprocess_input(np.expand_dims(image, axis = 0))

    def save(self, filepath):
        self.model = tf.keras.models.load_model(filepath) 
    

