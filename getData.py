from binance.client import Client
import pandas as pd
import datetime
import random
import numpy as np

tickers = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "AVAXUSDT", "SHIBUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT",
    "LINKUSDT", "XLMUSDT", "ETCUSDT", "FTMUSDT", "ALGOUSDT",
    "VETUSDT", "ICPUSDT", "FILUSDT", "THETAUSDT", "XMRUSDT",
    "EOSUSDT", "AAVEUSDT", "KSMUSDT", "GRTUSDT", "MANAUSDT",
    "SANDUSDT", "AXSUSDT", "CHZUSDT", "ZILUSDT", "ENJUSDT",
    "NEARUSDT", "LUNAUSDT", "WAVESUSDT", "DASHUSDT",
    "ZECUSDT", "QTUMUSDT", "BATUSDT", "COMPUSDT", "YFIUSDT",
    "SNXUSDT", "1INCHUSDT", "CRVUSDT", "SUSHIUSDT", "UMAUSDT",
    "BALUSDT", "RENUSDT", "HOTUSDT", "KAVAUSDT",
    "CELOUSDT", "STXUSDT", "ANKRUSDT", "LRCUSDT",
    "ZRXUSDT", "DCRUSDT", "SCUSDT", "XEMUSDT", "ONTUSDT",
    "RVNUSDT", "ICXUSDT", "ZENUSDT", "IOSTUSDT",
    "CVCUSDT", "STORJUSDT", "DGBUSDT", "BNTUSDT", "OCEANUSDT",
    "SXPUSDT", "FTTUSDT", "KNCUSDT", "GLMUSDT", "LPTUSDT",
    "ARUSDT", "CTKUSDT", "JSTUSDT", "AKROUSDT", "ROSEUSDT"
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
        
        if len(df) == 0:
            print(f"Skipping {symbol} - no data available")
            return None

        df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']] = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']].pct_change().fillna(0)
        df["timestamp"] = df["timestamp"].astype(int) / 10**9
        df = df.dropna().reset_index(drop=True)

        df.to_csv(f"data/{symbol}_data.csv", index=False)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    
def load_samples(batches = 1024, num_of_assets=10, window=64, mode='train', val_asset_ratio=0.2, seed=42):
    """
    Load training/validation samples from historical crypto data
    
    Args:
        batches: Number of samples to generate
        num_of_assets: Number of assets to use
        window: Lookback window size
        mode: 'train' or 'val' - train uses training assets, val uses validation assets
        val_asset_ratio: Fraction of total assets reserved for validation (0.2 = 20%)
        seed: Random seed for reproducible asset splits
    
    Note: Uses ALL available historical data for each asset since train/val split is done at asset level
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    available_tickers = [t for t in tickers]
    random.shuffle(available_tickers)
    
    num_val_assets = max(num_of_assets, int(len(available_tickers) * val_asset_ratio))
    num_train_assets = len(available_tickers) - num_val_assets
    
    train_tickers = available_tickers[:num_train_assets]
    val_tickers = available_tickers[num_train_assets:]
    
    if mode == 'train':
        if len(train_tickers) < num_of_assets:
            raise ValueError(f"Not enough training assets. Need {num_of_assets}, but only have {len(train_tickers)} available.")
        selected_tickers = random.sample(train_tickers, num_of_assets)
        print(f"Using TRAINING assets (100% of data): {', '.join(selected_tickers)}")
    else:
        if len(val_tickers) < num_of_assets:
            raise ValueError(f"Not enough validation assets. Need {num_of_assets}, but only have {len(val_tickers)} available. Increase total tickers or decrease num_of_assets.")
        selected_tickers = random.sample(val_tickers, num_of_assets)
        print(f"Using VALIDATION assets (100% of data, unseen during training): {', '.join(selected_tickers)}")
    
    data_frames = []
    missing_tickers = []
    for ticker in selected_tickers:
        try:
            df = pd.read_csv(f"data/{ticker}_data.csv")
            data_frames.append(df)
        except FileNotFoundError:
            missing_tickers.append(ticker)
    
    if missing_tickers:
        raise ValueError(f"Missing data files for: {', '.join(missing_tickers)}. Please run fetch_historical_data first.")
    
    if not data_frames:
        raise ValueError("No data files found. Please run fetch_historical_data first.")
    
    min_length = min(len(df) for df in data_frames)
    
    if min_length < window + 1:
        raise ValueError(f"Not enough data. Need at least {window + 1} samples, but only have {min_length}")
    
    start_range = 0
    end_range = min_length - window - 1
    print(f"Using full dataset: {min_length} total samples, generating {batches} training batches")
    
    if end_range <= start_range:
        raise ValueError(f"Not enough data. Need more historical data.")
    
    features = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    X_samples = []
    Y_samples = []
    
    for _ in range(batches):
        start_idx = random.randint(start_range, end_range)
        
        X_batch = []
        Y_batch = []
        
        for df in data_frames:
            window_data = df.iloc[start_idx:start_idx + window][features].values
            X_batch.append(window_data)
            
            next_close_change = df.iloc[start_idx + window]['close']
            Y_batch.append(next_close_change)
        
        X_sample = np.array(X_batch)
        Y_sample = np.array(Y_batch)
        
        X_samples.append(X_sample)
        Y_samples.append(Y_sample)
    
    X = np.array(X_samples)
    Y = np.array(Y_samples)
    
    X = X.reshape(batches, len(data_frames), window, -1)
    
    return X, Y