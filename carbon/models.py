# from django.db import models

# # Create your models here.
# import torch

# # Define Carbon Emission Estimation Model
# class CarbonEstimator(torch.nn.Module):
#     def __init__(self):
#         super(CarbonEstimator, self).__init__()
#         self.linear = torch.nn.Linear(4, 1)  # 4 input features, 1 output
    
#     def forward(self, x):
#         return self.linear(x)

# # Function to load model with trained weights
# def load_model():
#     model = CarbonEstimator()
#     model.load_state_dict(torch.load('carbon_model.pth', map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # Function to predict carbon emission
# def predict_carbon_emission(gas_used, gas_price, tx_type, token_standard):
#     model = load_model()
#     input_tensor = torch.tensor([[gas_used, gas_price, tx_type, token_standard]], dtype=torch.float32)

#     with torch.no_grad():
#         prediction = model(input_tensor).item()

#     # Debugging print
#     print(f"Input: {[gas_used, gas_price, tx_type, token_standard]}, Prediction: {prediction}")

#     return max(0, prediction)  # Ensure non-negative output
