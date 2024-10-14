# stock_predictor.py

import joblib
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np


class StockPredictor:
    def __init__(self, model_path='best_model.joblib'):
        self.model = joblib.load(model_path)
        self.tickers = ["^DJI", "^IXIC", "^GSPC", "^RUT", "^TYX", "GC=F", "SI=F", "BTC-USD", "ETH-USD", "AAPL", "AMZN",
                        "GOOG", "META", "MSFT", "NVDA", "TSLA"]


    def predict(self):
        data = pd.read_csv("feats.csv")
        predictions = self.model.predict(data)

        # Create a DataFrame with predictions
        pred_df = pd.DataFrame(predictions, columns=[f'Change_{i + 1}-Day' for i in range(predictions.shape[1])])
        pred_df['Ticker'] = data.iloc[:, :len(self.tickers)].idxmax(axis=1)

        # Save predictions to CSV
        pred_df.to_csv('predictions.csv', index=False)

        # Save predictions to a readable text file
        with open('predictions.txt', 'w') as f:
            for _, row in pred_df.iterrows():
                f.write(f"Ticker: {row['Ticker']}\n")
                for i in range(1, predictions.shape[1] + 1):
                    f.write(f"  {i}-Day Change: {row[f'Change_{i}-Day']:.2%}\n")
                f.write("\n")

        print("Predictions saved to 'predictions.csv' and 'predictions.txt'")


if __name__ == '__main__':
    predictor = StockPredictor()
    predictor.predict()