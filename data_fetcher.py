import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataFetcher:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self, ticker, period="2y", interval="1d"):
        """
        Fetch stock data using yfinance
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        stock_data = yf.download(ticker, period=period, interval=interval)
        return stock_data
    
    def prepare_data(self, data, sequence_length=60, train_split=0.8):
        """
        Prepare data for LSTM model
        
        Args:
            data (pd.DataFrame): Stock data
            sequence_length (int): Number of previous time steps to use as input features
            train_split (float): Ratio of data to use for training
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        # Focus on closing prices and convert to numpy array
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def prepare_features(self, data):
        """
        Create additional technical features from stock data
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        df = data.copy()
        
        # Calculate technical indicators
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['SD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['20MA'] + (df['SD'] * 2)
        df['Lower_Band'] = df['20MA'] - (df['SD'] * 2)
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate log returns
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Drop NaN values
        df = df.dropna()
        
        return df 