from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import json
import pandas as pd

class model():
    
    data_path = "data.json"
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        # Load data from a JSON file
        with open(self.data_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Ensure the index is numeric and convert to datetime
        df.index = pd.to_numeric(df.index)
        df.index = pd.to_datetime(df.index, unit="ms")
        
        return df
    
    def preprocess_data(self, df):
    # Prepare features and target variable
        
        X = df[['Open', 'High', 'Low']]  # Adjusting to only include available columns
        y = df['Close']
        
        # Split the data into training and test sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        
    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        
    def evaluate(self):
        # Make predictions and evaluate the model
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse
    
    
    def predict(self, new_data):
        # Ensure new_data matches the features used for training
        new_data_scaled = self.scaler.transform(new_data[['Open', 'High', 'Low']])
        predicted_closing_price = self.model.predict(new_data_scaled)
        return predicted_closing_price