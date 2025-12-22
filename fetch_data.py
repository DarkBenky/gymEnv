#!/usr/bin/env python
"""
Script to fetch historical cryptocurrency data from Binance.
Run this before training the model.
"""

from getData import fetch_historical_data, tickers
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Fetch data for all tickers
# Using 1h interval for faster downloads, you can change to '1d', '4h', etc.
interval = '1h'
years = 2  # Fetch 2 years of data

print(f"Fetching {years} years of {interval} data for {len(tickers)} cryptocurrencies...")
print("This may take a few minutes...\n")

successful = 0
failed = 0

for ticker in tickers:
    result = fetch_historical_data(ticker, interval, years=years)
    if result is not None:
        successful += 1
    else:
        failed += 1
    print()  # Empty line for readability

print(f"\nData fetch complete!")
print(f"Successfully fetched: {successful}/{len(tickers)}")
print(f"Failed: {failed}/{len(tickers)}")

if successful > 0:
    print(f"\nData saved in 'data/' directory")
    print("You can now run 'python env.py' to train the model")
else:
    print("\nNo data was fetched. Please check your internet connection.")
