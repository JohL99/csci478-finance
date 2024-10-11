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

def saveDataFrameToCSV(df, file_name):
    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=True)  # Set index=True to keep the datetime index in the file
    print(f"DataFrame saved to {file_name}")



if __name__ == "__main__":
    print("starting...\n")
    
    # Get data from yfinance and save it to a JSON file
    fetchAndSaveData()
    
    # Create an instance of the RegressionModel
    reg_model = model(DATA_PATH)
    
    # Load and preprocess the data
    df = reg_model.load_data()
    reg_model.preprocess_data(df)
    
    # Train the model
    reg_model.train()
    
    # Evaluate the model
    mse = reg_model.evaluate()
    print(f"Mean Squared Error: {mse}\n")
    
    # Backtest the model to predict closing prices for the data
    df = reg_model.backtest(df)
    
    # Save the backtested data to CSV
    saveDataFrameToCSV(df[['Open', 'Close', 'test_close']], "backtested_data.csv")
    
    # Example of predicting the closing price using standardized features
    # Create a DataFrame with sample features including new ones (with placeholders for now)
    example_features = pd.DataFrame([[430, 435, 420, 431, 429, 0.002, 1.5]],
                                    columns=['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility'])
    
    predicted_closing_price = reg_model.predict(example_features)
    print(f"Predicted Closing Price: {predicted_closing_price[0]}\n")
    
    # Calculate and display the price change
    opening_price = example_features['Open'][0]
    predicted_closing_price_value = predicted_closing_price.item()
    price_change = abs(predicted_closing_price_value - opening_price)
    
    if predicted_closing_price_value < opening_price:
        print(f"Price went down: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}, Change = {price_change:.2f}")
    elif predicted_closing_price_value > opening_price:
        print(f"Price went up: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}, Change = {price_change:.2f}")
    else:
        print(f"No price change: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}")


