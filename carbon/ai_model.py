import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

class CarbonFootprintModel:
    def __init__(self):
        self.model = self.build_model()
        self.scaler = None  # Scaler will be fit only on training data
        self.ENERGY_PER_GAS = 2.06e-5  # kWh per gas unit
        self.CARBON_INTENSITY = 0.283  # kg CO2 per kWh
        
        # Transaction type mappings
        self.TX_TYPE_MAPPING = {
            "ERC-20 Transfer": 1,
            "ERC-721 Transfer": 2,
            "ERC-721 Mint": 3,
            "ERC-1155 Transfer": 4,
            "ERC-1155 Mint": 5,
            "DeFi Swap": 6,
            "DeFi Liquidity Addition": 7,
            "Staking Transaction": 8
        }

        self.TOKEN_STANDARD_MAPPING = {
            "ERC-20": 1,
            "ERC-721": 2,
            "ERC-1155": 3
        }

        self.METHOD_ID_MAPPING = {
            "0xa9059cbb": "ERC-20 Transfer",
            "0x23b872dd": "ERC-721 Transfer",
            "0x40c10f19": "ERC-721 Mint",
            "0xf242432a": "ERC-1155 Transfer",
            "0x2eb2c2d6": "ERC-1155 Mint",
            "0x38ed1739": "DeFi Swap",
            "0xe8e33700": "DeFi Liquidity Addition",
            "0xa694fc3a": "Staking Transaction"
        }

        # Method type impact factors
        self.METHOD_TYPE_FACTORS = {
            0: 1.0,    # Unknown/Basic transfer
            1: 1.1,    # ERC-20 Transfer
            2: 1.2,    # ERC-721 Transfer
            3: 1.3,    # ERC-721 Mint
            4: 1.2,    # ERC-1155 Transfer
            5: 1.3,    # ERC-1155 Mint
            6: 1.4,    # DeFi Swap
            7: 1.5,    # DeFi Liquidity Addition
            8: 1.2     # Staking Transaction
        }
        
    def build_model(self):
        model = Sequential([
            Dense(256, input_dim=6, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        return model

    def calculate_carbon_footprint(self, gas_used, gas_price, method_type=0, input_data_len=0):
        """
        Enhanced carbon footprint calculation
        """
        # Base energy consumption
        base_energy = gas_used * self.ENERGY_PER_GAS
        base_carbon = base_energy * self.CARBON_INTENSITY
        
        # Network congestion factor
        gas_price_gwei = gas_price / 1e9
        network_congestion_factor = math.log1p(gas_price_gwei) / math.log1p(1000)
        
        # Computational complexity factor
        complexity_factor = 1 + (math.log1p(input_data_len) / math.log1p(1000))
        
        # Transaction type impact
        type_factor = self.METHOD_TYPE_FACTORS.get(method_type, 1.0)
        
        # Combine all factors
        total_carbon = (
            base_carbon * 
            (1 + (network_congestion_factor * 0.5)) *
            complexity_factor * 
            type_factor
        )
        
        # Apply bounds
        min_carbon = base_carbon * 0.8
        max_carbon = base_carbon * 4.0
        
        return np.clip(total_carbon, min_carbon, max_carbon)

    def preprocess_data(self, transactions, fit_scaler=False):
        if not transactions:
            return np.array([]).reshape(0, 6)
        
        features = []
        for tx in transactions:
            gas_used = float(tx['gas'])
            gas_price = float(tx['gasPrice'])
            
            # Calculate baseline carbon
            energy = gas_used * self.ENERGY_PER_GAS
            baseline_carbon = energy * self.CARBON_INTENSITY
            
            # Encoding input data
            input_data_len = len(tx['input']) / 2 - 1
            tx_type = 1 if tx['input'] == "0x" else 0
            
            # Determine method type
            method_type = self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")
            if method_type != "Unknown":
                method_type = list(self.METHOD_ID_MAPPING.values()).index(method_type) + 1
            else:
                method_type = 0
                
            features.append([gas_used, gas_price / 1e9, input_data_len, tx_type, method_type, baseline_carbon])
        
        features = np.array(features)
        if len(features) > 0:
            if fit_scaler:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)
        
        return features

    def predict_emissions_for_transactions(self, transactions):
        if not transactions:
            return []
        
        if self.scaler is None:
            features = self.preprocess_data(transactions, fit_scaler=True)
        else:
            features = self.preprocess_data(transactions, fit_scaler=False)
        
        if len(features) == 0:
            return []
        
        try:
            predictions = self.model.predict(features, verbose=0)
        except Exception as e:
            print(f"Prediction error: {e}")
            predictions = []
            for tx in transactions:
                gas_used = float(tx['gas'])
                gas_price = float(tx['gasPrice'])
                method_type = self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")
                if method_type != "Unknown":
                    method_type = list(self.METHOD_ID_MAPPING.values()).index(method_type) + 1
                else:
                    method_type = 0
                input_data_len = len(tx['input']) / 2 - 1
                
                carbon_footprint = self.calculate_carbon_footprint(
                    gas_used=gas_used,
                    gas_price=gas_price,
                    method_type=method_type,
                    input_data_len=input_data_len
                )
                predictions.append([carbon_footprint])
        
        emissions = []
        for i, tx in enumerate(transactions):
            if len(predictions) > i:
                model_prediction = max(predictions[i][0], 0)
                # Calculate baseline for comparison
                baseline = self.calculate_carbon_footprint(
                    gas_used=float(tx['gas']),
                    gas_price=float(tx['gasPrice']),
                    method_type=list(self.METHOD_ID_MAPPING.values()).index(self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")) + 1 if tx['methodId'] in self.METHOD_ID_MAPPING else 0,
                    input_data_len=len(tx['input']) / 2 - 1
                )
                final_emission = (model_prediction + baseline) / 2
            else:
                final_emission = self.calculate_carbon_footprint(
                    gas_used=float(tx['gas']),
                    gas_price=float(tx['gasPrice']),
                    method_type=list(self.METHOD_ID_MAPPING.values()).index(self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")) + 1 if tx['methodId'] in self.METHOD_ID_MAPPING else 0,
                    input_data_len=len(tx['input']) / 2 - 1
                )
            
            emissions.append({
                'transaction_hash': tx.get('hash', 'UNKNOWN_TX'),
                'predicted_carbon_emission': final_emission
            })
        
        return emissions

    def train(self, transactions, epochs=100):
        if not transactions:
            return None
        
        features = self.preprocess_data(transactions, fit_scaler=True)
        if len(features) == 0:
            return None
        
        labels = []
        for tx in transactions:
            gas_used = float(tx['gas'])
            gas_price = float(tx['gasPrice'])
            method_type = self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")
            if method_type != "Unknown":
                method_type = list(self.METHOD_ID_MAPPING.values()).index(method_type) + 1
            else:
                method_type = 0
            input_data_len = len(tx['input']) / 2 - 1
            
            # Use enhanced carbon footprint calculation
            carbon_footprint = self.calculate_carbon_footprint(
                gas_used=gas_used,
                gas_price=gas_price,
                method_type=method_type,
                input_data_len=input_data_len
            )
            labels.append(carbon_footprint)
        
        labels = np.array(labels)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        history = self.model.fit(
            features,
            labels,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import math

# class CarbonFootprintModel:
#     def __init__(self):
#         self.model = self.build_model()
#         self.scaler = None  # Scaler will be fit only on training data
#         self.ENERGY_PER_GAS = 2.06e-5  # kWh per gas unit
#         self.CARBON_INTENSITY = 0.283  # kg CO2 per kWh
#         self.TX_TYPE_MAPPING = {
#             "ERC-20 Transfer": 1,
#             "ERC-721 Transfer": 2,
#             "ERC-721 Mint": 3,
#             "ERC-1155 Transfer": 4,
#             "ERC-1155 Mint": 5,
#             "DeFi Swap": 6,
#             "DeFi Liquidity Addition": 7,
#             "Staking Transaction": 8
#         }

#         self.TOKEN_STANDARD_MAPPING = {
#             "ERC-20": 1,
#             "ERC-721": 2,
#             "ERC-1155": 3
#         }

#         self.METHOD_ID_MAPPING = {
#             "0xa9059cbb": "ERC-20 Transfer",
#             "0x23b872dd": "ERC-721 Transfer",
#             "0x40c10f19": "ERC-721 Mint",
#             "0xf242432a": "ERC-1155 Transfer",
#             "0x2eb2c2d6": "ERC-1155 Mint",
#             "0x38ed1739": "DeFi Swap",
#             "0xe8e33700": "DeFi Liquidity Addition",
#             "0xa694fc3a": "Staking Transaction"
#         }
        
#     def build_model(self):
#         model = Sequential([
#             Dense(256, input_dim=6, activation='relu'),
#             BatchNormalization(),
#             Dropout(0.2),
#             Dense(128, activation='relu'),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(64, activation='relu'),
#             BatchNormalization(),
#             Dense(1, activation='linear')
#         ])
        
#         optimizer = Adam(learning_rate=0.0005)
#         model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
#         return model

#     def preprocess_data(self, transactions, fit_scaler=False):
#         if not transactions:
#             return np.array([]).reshape(0, 6)
        
#         features = []
#         for tx in transactions:
#             gas_used = float(tx['gas'])
#             gas_price = float(tx['gasPrice'])
            
#             # Calculate baseline carbon
#             energy = gas_used * self.ENERGY_PER_GAS
#             baseline_carbon = energy * self.CARBON_INTENSITY
            
#             # Encoding input data
#             input_data_len = len(tx['input']) / 2 - 1
#             tx_type = 1 if tx['input'] == "0x" else 0  # Simplified condition for transaction type
            
#             # Determine method type from methodId, default to "Unknown" if not found
#             method_type = self.METHOD_ID_MAPPING.get(tx['methodId'], "Unknown")
            
#             # Encode the method type as a numerical value (if necessary)
#             if method_type != "Unknown":
#                 method_type = list(self.METHOD_ID_MAPPING.values()).index(method_type) + 1  # 1-based index
#             else:
#                 method_type = 0  # Represent "Unknown" as 0
#             features.append([gas_used, gas_price / 1e9, input_data_len, tx_type, method_type, baseline_carbon])
        
#         features = np.array(features)
#         if len(features) > 0:
#             if fit_scaler:
#                 self.scaler = StandardScaler()
#                 features = self.scaler.fit_transform(features)
#             else:
#                 features = self.scaler.transform(features)
        
#         return features

#     def predict_emissions_for_transactions(self, transactions):
#         if not transactions:
#             return []
        
#         # Ensure scaler is fitted if not already
#         if self.scaler is None:
#             features = self.preprocess_data(transactions, fit_scaler=True)
#         else:
#             features = self.preprocess_data(transactions, fit_scaler=False)
        
#         if len(features) == 0:
#             return []
        
#         try:
#             predictions = self.model.predict(features, verbose=0)
#         except Exception as e:
#             print(f"Prediction error: {e}")
#             predictions = []
#             for tx in transactions:
#                 gas_used = float(tx['gas'])
#                 gas_price = float(tx['gasPrice'])
#                 energy = gas_used * self.ENERGY_PER_GAS
#                 predictions.append([energy * self.CARBON_INTENSITY])
        
#         emissions = []
#         for i, tx in enumerate(transactions):
#             gas_used = float(tx['gas'])
#             gas_price = float(tx['gasPrice'])
            
#             energy = gas_used * self.ENERGY_PER_GAS
#             baseline_carbon = energy * self.CARBON_INTENSITY
            
#             if len(predictions) > i:
#                 model_prediction = max(predictions[i][0], 0)
#                 final_emission = (model_prediction + baseline_carbon) / 2
#             else:
#                 final_emission = baseline_carbon
            
#             emissions.append({
#                 'transaction_hash': tx.get('hash', 'UNKNOWN_TX'),
#                 'predicted_carbon_emission': final_emission
#             })
        
#         return emissions


#     def train(self, transactions, epochs=100):
#         if not transactions:
#             return None
        
#         features = self.preprocess_data(transactions, fit_scaler=True)
#         if len(features) == 0:
#             return None
        
#         labels = []
#         for tx in transactions:
#             gas_used = float(tx['gas'])
#             gas_price = float(tx['gasPrice'])
            
#             energy = gas_used * self.ENERGY_PER_GAS
#             carbon = energy * self.CARBON_INTENSITY
#             gas_price_gwei = gas_price / 1e9
#             complexity_factor = math.log1p(gas_price_gwei) / 10  # Adjusted complexity factor
            
#             carbon_footprint = carbon * (1 + complexity_factor)
#             labels.append(carbon_footprint)
        
#         labels = np.array(labels)
        
#         early_stopping = EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True,
#             min_delta=0.0001
#         )
        
#         history = self.model.fit(
#             features,
#             labels,
#             epochs=epochs,
#             batch_size=32,
#             validation_split=0.2,
#             callbacks=[early_stopping],
#             verbose=1
#         )
        
#         return history

