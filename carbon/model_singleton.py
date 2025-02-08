import os
import joblib
from tensorflow.keras.models import load_model
from .ai_model import CarbonFootprintModel

class ModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = 'saved_model/carbon_footprint_model'
        self.scaler_path = 'saved_model/scaler.pkl'
        self.load_or_initialize_model()
    
    def load_or_initialize_model(self):
        """
        Load existing model and scaler, or initialize a new one if missing.
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                print("Loading existing model and scaler...")
                self.model = CarbonFootprintModel()
                self.model.model = load_model(self.model_path)
                self.model.scaler = joblib.load(self.scaler_path)
                
                # 🔹 Debugging: Check if scaler is loaded
                if self.model.scaler is None:
                    print("Scaler loading failed! Reinitializing it.")
                    self.model.scaler = self.initialize_scaler()
            else:
                print("No saved model found. Initializing new model...")
                self.initialize_new_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing a new model as fallback...")
            self.initialize_new_model()

    
    def initialize_new_model(self):
        """
        Initializes a new model without training.
        """
        self.model = CarbonFootprintModel()
        os.makedirs('saved_model', exist_ok=True)  # Ensure directory exists

    def train_model(self, training_data):
        """
        Retrain the model and save updated weights.
        """
        print("Training model with new data...")
        history = self.model.train(training_data)
        if history:
            self.model.model.save(self.model_path)
            joblib.dump(self.model.scaler, self.scaler_path)
            print("Model and scaler updated and saved.")
        return history
    
    def predict(self, transactions):
        """
        Make predictions using the trained model.
        """
        if self.model is None or self.model.model is None:
            print("Model not loaded! Returning default values.")
            return None
        return self.model.predict_emissions_for_transactions(transactions)


# import os
# from tensorflow.keras.models import load_model, save_model
# import joblib
# from .ai_model import CarbonFootprintModel

# class ModelSingleton:
#     _instance = None
    
#     @classmethod
#     def get_instance(cls):
#         if cls._instance is None:
#             cls._instance = cls()
#         return cls._instance
    
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.model_path = 'saved_model/carbon_footprint_model'
#         self.scaler_path = 'saved_model/scaler.pkl'
#         self.load_or_train_model()
    
#     def load_or_train_model(self):
#         try:
#             # Try to load existing model and scaler
#             if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
#                 print("Loading existing model and scaler...")
#                 self.model = CarbonFootprintModel()
#                 self.model.model = load_model(self.model_path)
#                 self.model.scaler = joblib.load(self.scaler_path)
#             else:
#                 print("Training new model...")
#                 # Initialize and train model with initial data
#                 self.model = CarbonFootprintModel()
#                 # You'll need to provide initial training data here
#                 # self.train_model(initial_training_data)
                
#                 # Create directory if it doesn't exist
#                 os.makedirs('saved_model', exist_ok=True)
                
#                 # Save the model and scaler
#                 self.model.model.save(self.model_path)
#                 joblib.dump(self.model.scaler, self.scaler_path)
                
#         except Exception as e:
#             print(f"Error loading/training model: {e}")
#             self.model = CarbonFootprintModel()  # Fallback to untrained model
    
#     def train_model(self, training_data):
#         """
#         Method to retrain the model if needed
#         """
#         history = self.model.train(training_data)
#         if history:
#             # Save the updated model and scaler
#             self.model.model.save(self.model_path)
#             joblib.dump(self.model.scaler, self.scaler_path)
#         return history
    
#     def predict(self, transactions):
#         """
#         Method to make predictions using the trained model
#         """
#         return self.model.predict_emissions_for_transactions(transactions)