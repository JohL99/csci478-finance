import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from model import model


DATA_PATH = "data.json"
ticker = "MSFT"

def fetchAndSaveData():
    msft = yf.Ticker(ticker)
    msftHist = msft.history(period="max")

    # Filter for 'Open', 'High', 'Low', 'Close' columns only
    selectedColumns = ['Open', 'High', 'Low', 'Close']
    msftHistFiltered = msftHist[selectedColumns]

    # Save the filtered data to JSON
    msftHistFiltered.to_json(DATA_PATH)

    return msftHistFiltered


if __name__ == "__main__":
    print("starting...\n\n")
    
    # get data from yfinance and save it to a JSON file
    fetchAndSaveData()
    data_path = "data.json"
    
    # Create an instance of the RegressionModel
    reg_model = model(data_path)
    
    # Load and preprocess the data
    df = reg_model.load_data()
    reg_model.preprocess_data(df)
    
    # Train the model
    reg_model.train()
    
    # Evaluate the model
    mse = reg_model.evaluate()
    print(f"Mean Squared Error: {mse}")
    
    # Example of predicting the closing price using standardized features
    example_features = pd.DataFrame([[430, 435, 420, 1000000]], columns=['Open', 'High', 'Low', 'Volume'])  # Example features
    predicted_closing_price = reg_model.predict(example_features)
    print(f"Predicted Closing Price: {predicted_closing_price[0]}")
    
