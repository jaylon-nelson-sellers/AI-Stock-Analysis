import warnings
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler
import yfinance as yf
from ta import add_all_ta_features
from pathlib import Path


class CreateStockData:
        
    def __init__(self, observation_days: int, target_days: int, tickers: list,
                 add_technical_indicators: bool = True):
        self.observation_days = observation_days
        self.target_days = target_days
        self.tickers = tickers
        self.add_ta = add_technical_indicators

        self.base_dir = Path(
            f"Data_d-{self.observation_days}_t-{self.target_days}")
    
    def get_recent_data(self):
        stock_data = self.get_stock_data()

        regression_targets = self.calculate_targets(stock_data['Close'], self.target_days)



        stock_data = self.prepare_features(stock_data, self.observation_days)

        #assert stock_data.shape[0] == regression_targets.shape[0]

        stock_data.reset_index(inplace=True, drop=True)

        
        stock_data = stock_data.iloc[-1]
        return stock_data

    def process_stock_data(self):
        stock_data = self.get_btc_data()
        if self.add_ta:
                stock_data = self.add_technical_indicators(stock_data)

        regression_targets = self.calculate_targets(stock_data['Close'], self.target_days)

        
        scaler = RobustScaler()
        scaled_array = scaler.fit_transform(stock_data.values)

        # Convert back to DataFrame with original columns and index preserved
        stock_data = pd.DataFrame(scaled_array, columns=stock_data.columns, index=stock_data.index)

        stock_data = stock_data.iloc[:-self.target_days]

        stock_data = self.prepare_features(stock_data, self.observation_days)

        assert stock_data.shape[0] == regression_targets.shape[0]

        stock_data.reset_index(inplace=True, drop=True)

        
        stock_data.iloc

        self.save_data(stock_data, regression_targets)




    def get_stock_data(self) -> pd.DataFrame:
        data_frames = []
        for ticker in self.tickers:
            data = self.download_stock_data(ticker)
            if self.add_ta:
                data = self.add_technical_indicators(data)
            data.index = self.convert_datetime_to_int(data.index)
            # Create a DataFrame with one-hot encoded column for this ticker
            data = pd.concat([data], axis=1)
            data_frames.append(data)

        combined_data = pd.concat(data_frames, axis=0).reset_index()

        # Identify and list columns that contain only zeros and are thus non-informative
        columns_to_delete = [col for col in combined_data.columns if combined_data[col].sum() == 0]

        # Remove the identified non-informative columns from the DataFrame
        stock_data_cleaned = combined_data.drop(columns=columns_to_delete)
        ########################################
        #remove date
        #stock_data_cleaned = stock_data_cleaned.drop('Datetime', axis=1)
        return stock_data_cleaned
    
    def get_btc_data(self) -> pd.DataFrame:
        data_frames = []
        combined_data = pd.read_csv("BTC_RAW.csv")
        combined_data.reset_index(inplace=True,drop=True)

        # Identify and list columns that contain only zeros and are thus non-informative
        columns_to_delete = [col for col in combined_data.columns if combined_data[col].sum() == 0]

        # Remove the identified non-informative columns from the DataFrame
        stock_data_cleaned = combined_data.drop(columns=columns_to_delete)
        ########################################
        #remove date
        #stock_data_cleaned = stock_data_cleaned.drop('Datetime', axis=1)
        return stock_data_cleaned

    def prepare_features(self, stock_data: pd.DataFrame, observation_days: int) -> pd.DataFrame:
        if observation_days == 1:
            return stock_data.iloc[observation_days:]

        combined_data = stock_data.copy()
        for day in range(observation_days, 0, -1):
            shifted_df = stock_data.shift(day)
            shifted_df.columns = [f"{col}_shifted_{day}" for col in stock_data.columns]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in stock_data.columns:
                    combined_data.insert(combined_data.columns.get_loc(col) + 1, f"{col}_shifted_{day}",
                                         shifted_df[f"{col}_shifted_{day}"])

        return combined_data[observation_days:]

    def calculate_targets(self, close_prices: pd.Series, future_days: int) -> pd.DataFrame:
        regression_targets = pd.DataFrame(index=close_prices.index)

        for day in range(1,future_days + 1):
            future_close = close_prices.shift(-day)

            # Calculate the  change for the regression target
            regression_targets[f"Change_{day}-Day"] = future_close

        return regression_targets.iloc[self.observation_days:-self.target_days]

    def save_data(self, features: pd.DataFrame, regress_targets: pd.DataFrame) -> int:
        #self.base_dir.mkdir(parents=True, exist_ok=True)

        features_path = str(self.tickers[0]) +'_feats.csv'
        regress_targets_path =str(self.tickers[0]) + '_regress.csv'

        features.fillna(0).to_csv(features_path, index=False)
        regress_targets.fillna(0).to_csv(regress_targets_path, index=False)

        return 0

    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        y = yf.Ticker(symbol)
        hist = y.history(interval="1D", period="max")
        hist = hist.drop(columns=['Dividends', 'Stock Splits'], errors='ignore').dropna()
        return hist

    @staticmethod
    def add_technical_indicators(df):
        df_cleaned = df.fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_with_indicators = add_all_ta_features(
                df_cleaned, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        return df_with_indicators

    @staticmethod
    def convert_datetime_to_int(dates):
        return dates.astype('int64') // 10 ** 9


if __name__ == '__main__':
    num_stocks = 1
    tickers = [
    'BTC-USD',
    ]

    #24 = 2 hours
    St = CreateStockData(1, 12, tickers, add_technical_indicators=True)
    St.process_stock_data()