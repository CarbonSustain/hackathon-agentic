import requests
from fastapi import FastAPI
from web3 import Web3

app = FastAPI()

INFURA_URL = "https://mainnet.infura.io/v3/3181336eb56b483a8d1677e94ce2906c"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
ETHERSCAN_API_KEY = "TQA3BPKKY7PFFZJJ3MKCHPRI8CKVFVTCXQ"

def estimate_carbon_footprint(tx):
    gas_used = int(tx.get("gasUsed", "0"))
    gas_price = int(tx.get("gasPrice", "0"))
    
    # Estimate carbon footprint using gas consumption (Example: 0.0000001 CO2 per gas unit)
    carbon_emission = gas_used * 0.0000001  

    if "transfer" in tx.get("input", ""):
        carbon_emission += 0.1 
    elif "mint" in tx.get("input", ""):
        carbon_emission += 5.0  
    
    return round(carbon_emission, 6)

@app.get("/transactions/{address}")
def get_transactions(address: str):
    if not w3.is_address(address):
        return {"error": "Invalid Ethereum address"}

    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": "Failed to fetch transactions"}

    data = response.json()
    if data["status"] != "1":
        return {"error": "No transactions found"}

    transactions = data["result"]
    for tx in transactions:
        tx["carbon_footprint"] = estimate_carbon_footprint(tx)

    return {"address": address, "transactions": transactions}
