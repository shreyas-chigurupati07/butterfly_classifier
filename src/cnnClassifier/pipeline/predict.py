import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

class PredictionPipeline:
    def __init__(self, filename, model_path='artifacts/training/model.h5',
                 class_mapping_path='artifacts/data_ingestion/data/class_mapping.json'):
        self.filename = filename
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path

    def load_class_mapping(self):
        """Load class index-to-label mapping from a JSON file."""
        if os.path.exists(self.class_mapping_path):
            with open(self.class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
                return {str(v): k for k, v in class_mapping.items()}
        else:
            print("⚠️ Warning: Class mapping file not found. Using index values.")
            return None

    def preprocess_image(self, img_path):
        """Preprocess input image for model prediction."""
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to VGG input size
        img = image.img_to_array(img)  # Convert to array
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize (rescale pixel values)
        return img

    def predict(self):
        """Predict the class of the given image."""
        # Load the trained model
        model = load_model(self.model_path)

        # Preprocess the input image
        test_image = self.preprocess_image(self.filename)

        # Make prediction
        result_index = np.argmax(model.predict(test_image), axis=1)[0]  # Get predicted class index

        # Load class labels
        class_mapping = self.load_class_mapping()

        # Get class label from mapping
        if class_mapping and str(result_index) in class_mapping:
            prediction_label = class_mapping[str(result_index)]
        else:
            prediction_label = f"Class {result_index}"  # Fallback if mapping is missing

        print(f"✅ Predicted Class Index: {result_index} → Label: {prediction_label}")
        return {"image": self.filename, "prediction": prediction_label}















# import numpy as np
# import tensorflow as tf
# import h5py
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os



# class PredictionPipeline:
#     def __init__(self, filename):
#         self.filename = filename



#     def predict(self):
#         model = load_model(os.path.join('artifacts', 'training', 'model.h5'))


#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size = (224, 224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         if result[0] == 1:
#             prediction = 'Healthy'
#             return[{ "image" : prediction}]
        
#         else:
#             prediction = 'Diseased'
#             return[{ "image" : prediction}]