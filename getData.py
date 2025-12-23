from binance.client import Client
import pandas as pd
import datetime
import random
import numpy as np

tickers = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "AVAXUSDT", "SHIBUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT",
    "LINKUSDT", "XLMUSDT", "ETCUSDT", "FTMUSDT", "ALGOUSDT"
]

def fetch_historical_data(symbol, interval, years=3):
    try:
        client = Client()

        print(f"Fetching data for {symbol}...")
    
        klines = client.get_historical_klines(symbol, interval, start_str=f"{years} year ago UTC")

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])


        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.astype({
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "float64",
            "quote_asset_volume": "float64", "number_of_trades": "int64",
            "taker_buy_base_asset_volume": "float64",
            "taker_buy_quote_asset_volume": "float64"
        })
        print(f"Successfully fetched {len(df)} records for {symbol}")

        # normalize the values as pct change
        df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']] = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']].pct_change().fillna(0)
        df["timestamp"] = df["timestamp"].astype(int) / 10**9  # convert to seconds
        df = df.dropna().reset_index(drop=True)

        # Save to CSV
        df.to_csv(f"data/{symbol}_data.csv", index=False)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    
def load_samples(batches = 1024, num_of_assets=10, window=64, train_split=0.8, mode='train'):
    """
    Return X n samples of window_size*num_of_assets*batches shape
    Returns Y number_of_assets*batches shape (next value close change)
    
    Args:
        batches: Number of samples to generate
        num_of_assets: Number of crypto assets to use
        window: Size of the lookback window
        train_split: Fraction of data to use for training (0.8 = 80% train, 20% test)
        mode: 'train' or 'test' - determines which portion of data to use
    """
    
    # Select random n tickers from the available tickers list
    selected_tickers = random.sample(tickers, min(num_of_assets, len(tickers)))
    
    # Load data for selected assets
    data_frames = []
    for ticker in selected_tickers:
        try:
            df = pd.read_csv(f"data/{ticker}_data.csv")
            data_frames.append(df)
        except FileNotFoundError:
            print(f"Warning: Data file for {ticker} not found. Skipping.")
            continue
    
    if not data_frames:
        raise ValueError("No data files found. Please run fetch_historical_data first.")
    
    # Get the minimum length to ensure all assets have enough data
    min_length = min(len(df) for df in data_frames)
    
    if min_length < window + 1:
        raise ValueError(f"Not enough data. Need at least {window + 1} samples, but only have {min_length}")
    
    # Calculate train/test split indices
    split_idx = int(min_length * train_split)
    
    if mode == 'train':
        start_range = 0
        end_range = split_idx - window - 1
        print(f"Loading TRAINING data: indices 0 to {split_idx}")
    else:  # test mode
        start_range = split_idx
        end_range = min_length - window - 1
        print(f"Loading TEST data: indices {split_idx} to {min_length}")
    
    if end_range <= start_range:
        raise ValueError(f"Not enough data for {mode} set. Need more historical data.")
    
    # Features to use for X
    features = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    
    X_samples = []
    Y_samples = []
    
    for _ in range(batches):
        # Random starting point ONLY within the allowed range (train or test)
        # Same start_idx for ALL assets so windows are aligned in time
        start_idx = random.randint(start_range, end_range)
        
        # Collect window data for all assets using the SAME time window
        X_batch = []
        Y_batch = []
        
        for df in data_frames:
            # Get window of features for this asset at the same time period
            window_data = df.iloc[start_idx:start_idx + window][features].values
            X_batch.append(window_data)
            
            # Get next close price change as target (Y)
            next_close_change = df.iloc[start_idx + window]['close']
            Y_batch.append(next_close_change)
        
        # Stack all assets for this sample
        # Shape: (num_assets, window, num_features)
        X_sample = np.array(X_batch)
        Y_sample = np.array(Y_batch)
        
        X_samples.append(X_sample)
        Y_samples.append(Y_sample)
    
    # Convert to numpy arrays
    X = np.array(X_samples)  # Shape: (batches, num_assets, window, num_features)
    Y = np.array(Y_samples)  # Shape: (batches, num_assets)
    
    # Reshape X to (batches, window, num_assets * num_features) for easier processing
    X = X.reshape(batches, len(data_frames), window, -1)
    
    return X, Y