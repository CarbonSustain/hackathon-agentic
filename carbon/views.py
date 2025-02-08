from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from web3 import Web3
from .model_singleton import ModelSingleton

# Web3 Infura setup
INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Etherscan API Key
ETHERSCAN_API_KEY = "TQA3BPKKY7PFFZJJ3MKCHPRI8CKVFVTCXQ"

# Get model singleton instance
model_singleton = ModelSingleton.get_instance()

@api_view(['GET'])
def fetch_and_predict_carbon(request, address):
    if not w3.is_address(address):
        return Response({"error": "Invalid Ethereum address"}, status=400)

    # Fetch transactions from Etherscan
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return Response({"error": "Failed to fetch transactions from Etherscan"}, status=500)

    data = response.json()

    # Check API response
    if data.get("status") != "1":
        return Response({"error": f"API response error: {data.get('message', 'Unknown error')}"}, status=500)

    if not data.get("result", []):
        return Response({"error": "No transactions found for the given address"}, status=404)

    # Update METHOD_ID_MAPPING with new descriptions
    model_singleton.model.METHOD_ID_MAPPING = {
        "0xa9059cbb": "ERC-20 Token Transfer",
        "0x23b872dd": "ERC-721 NFT Transfer",
        "0x40c10f19": "ERC-721 NFT Mint",
        "0xf242432a": "ERC-1155 NFT Transfer",
        "0x2eb2c2d6": "ERC-1155 Batch NFT Mint",
        "0x38ed1739": "DeFi Swap",
        "0xe8e33700": "DeFi Liquidity Addition",
        "0xa694fc3a": "Staking Transaction"
    }

    transactions = data["result"]
    
    # Process transactions and predict carbon emissions
    processed_transactions = []
    for tx in transactions:
        gas_used = float(tx.get("gasUsed", 0))
        gas_price = float(tx.get("gasPrice", 0))
        method_id = tx.get("methodId", "")
        input_data = tx.get("input", "")
        to_address = tx.get("to", "")

        # Determine transaction type
        if input_data == "0x":
            tx_type = "ETH Transfer"
        else:
            # First check for known method IDs
            tx_type = model_singleton.model.METHOD_ID_MAPPING.get(method_id, "Smart Contract Interaction")

        # Predict carbon emission using singleton model instance
        carbon_emission = model_singleton.predict([{
            "gas": gas_used,
            "gasPrice": gas_price,
            "input": input_data,
            "methodId": method_id
        }])[0]['predicted_carbon_emission']

        # Keep original transaction data and add AI prediction
        tx["ai-carbon-footprint"] = carbon_emission
        tx["transaction_type"] = tx_type
        processed_transactions.append(tx)

    return Response({"address": address, "transactions": processed_transactions})

# from django.shortcuts import render
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# import requests
# from web3 import Web3
# from .ai_model import CarbonFootprintModel  # Import the AI model

# # Web3 Infura setup
# INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
# w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# # Etherscan API Key
# ETHERSCAN_API_KEY = "TQA3BPKKY7PFFZJJ3MKCHPRI8CKVFVTCXQ"


# @api_view(['GET'])
# def fetch_and_predict_carbon(request, address):
#     if not w3.is_address(address):
#         return Response({"error": "Invalid Ethereum address"}, status=400)

#     # Fetch transactions from Etherscan
#     url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={ETHERSCAN_API_KEY}"
#     response = requests.get(url)

#     if response.status_code != 200:
#         return Response({"error": "Failed to fetch transactions from Etherscan"}, status=500)

#     data = response.json()

#     # Check API response
#     if data.get("status") != "1":
#         return Response({"error": f"API response error: {data.get('message', 'Unknown error')}"}, status=500)

#     if not data.get("result", []):
#         return Response({"error": "No transactions found for the given address"}, status=404)

#     transactions = data["result"]
    
#     # Initialize AI model
#     model = CarbonFootprintModel()

#     # Process transactions and predict carbon emissions
#     processed_transactions = []
#     for tx in transactions:
#         gas_used = float(tx.get("gasUsed", 0))
#         gas_price = float(tx.get("gasPrice", 0))
#         method_id = tx.get("methodId", "")

#         # Determine transaction type
        

#         # Predict carbon emission using AI model
#         carbon_emission = model.predict_emissions_for_transactions([{
#             "gas": gas_used,
#             "gasPrice": gas_price,
#             "input": tx["input"],
#             "methodId": method_id
#         }])[0]['predicted_carbon_emission']

#         # Add AI prediction to transaction data
#         tx["ai-carbon-footprint"] = carbon_emission  # Convert to metric tons
#         processed_transactions.append(tx)

#     return Response({"address": address, "transactions": processed_transactions})




# from django.shortcuts import render
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# import torch
# import requests
# from web3 import Web3
# from .models import predict_carbon_emission

# # Web3 Infura setup
# INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
# w3 = Web3(Web3.HTTPProvider(INFURA_URL))

# # Etherscan API Key
# ETHERSCAN_API_KEY = "TQA3BPKKY7PFFZJJ3MKCHPRI8CKVFVTCXQ"

# # Transaction type mappings
# TX_TYPE_MAPPING = {
#     "ERC-20 Transfer": 1,
#     "ERC-721 Transfer": 2,
#     "ERC-721 Mint": 3,
#     "ERC-1155 Transfer": 4,
#     "ERC-1155 Mint": 5,
#     "DeFi Swap": 6,
#     "DeFi Liquidity Addition": 7,
#     "Staking Transaction": 8
# }

# TOKEN_STANDARD_MAPPING = {
#     "ERC-20": 1,
#     "ERC-721": 2,
#     "ERC-1155": 3
# }

# METHOD_ID_MAPPING = {
#     "0xa9059cbb": "ERC-20 Transfer",
#     "0x23b872dd": "ERC-721 Transfer",
#     "0x40c10f19": "ERC-721 Mint",
#     "0xf242432a": "ERC-1155 Transfer",
#     "0x2eb2c2d6": "ERC-1155 Mint",
#     "0x38ed1739": "DeFi Swap",
#     "0xe8e33700": "DeFi Liquidity Addition",
#     "0xa694fc3a": "Staking Transaction"
# }

# @api_view(['GET'])
# def fetch_and_predict_carbon(request, address):
#     if not w3.is_address(address):
#         return Response({"error": "Invalid Ethereum address"}, status=400)

#     # Fetch transactions from Etherscan
#     url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={ETHERSCAN_API_KEY}"
#     response = requests.get(url)

#     if response.status_code != 200:
#         return Response({"error": "Failed to fetch transactions from Etherscan"}, status=500)

#     data = response.json()

#     # Check the API response status and message
#     if data.get("status") != "1":
#         return Response({"error": f"API response error: {data.get('message', 'Unknown error')}"}, status=500)

#     # Check if transactions are found
#     if len(data.get("result", [])) == 0:
#         return Response({"error": "No transactions found for the given address"}, status=404)

#     transactions = data["result"]
    
#     # Process transactions and predict carbon emissions
#     for tx in transactions:
#         # Convert gasUsed and gasPrice to floats
#         gas_used = float(tx.get("gasUsed", 0))
#         gas_price = float(tx.get("gasPrice", 0))
#         method_id = tx.get("methodId", "")
        
#         # Default to zero if methodId is unrecognized
#         tx_type = TX_TYPE_MAPPING.get(METHOD_ID_MAPPING.get(method_id, ""), 0)
#         token_standard = 1 if "ERC-20" in METHOD_ID_MAPPING.get(method_id, "") else 2 if "ERC-721" in METHOD_ID_MAPPING.get(method_id, "") else 3
        
#         # AI model prediction
#         carbon_emission = predict_carbon_emission(gas_used, gas_price, tx_type, token_standard)
        
#         # Add AI carbon footprint prediction to transaction data
#         tx["ai-carbon-footprint"] = carbon_emission

#     return Response({"address": address, "transactions": transactions})

