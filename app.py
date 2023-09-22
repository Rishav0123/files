import pandas as pd
import yfinance as yf
import requests
from flask import Flask, render_template, jsonify, request
import numpy as np
import os
import openpyxl

# Define the path to your Excel file with a list of stock symbols (e.g., stocks.xlsx)
excel_file_path = 'stock_list.xlsx'
app = Flask(__name__)

def download_yahoo_data(symbol, period="365d"):
    try:
        # Set a timeout for the data download request (e.g., 10 seconds)
        data = yf.download(symbol, period=period, timeout=10)
        return data
    except requests.exceptions.Timeout:
        print(f"Data download for {symbol} timed out. Skipping...")
        return None
    except Exception as e:
        print(f"An error occurred while downloading data for {symbol}: {str(e)}")
        return None


def get_macd(data, slow_period=26, fast_period=12, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) and related indicators for a DataFrame.

    Parameters:
    - data: DataFrame containing price data (columns are different securities).
    - slow_period: Slow EMA period (default is 26).
    - fast_period: Fast EMA period (default is 12).
    - signal_period: Signal line period (default is 9).

    Returns:
    - DataFrame with columns 'MACD', 'Signal', and 'Histogram' for each security.
    """
    # Create an empty list to store dataframes for each stock symbol
    dataframes = []

# Create an empty DataFrame to store MACD data
    #macd_df = pd.DataFrame()
    macd_data = pd.DataFrame()

    # Loop through each column (security) in the DataFrame
    for column in data.columns:
        # Calculate fast and slow Exponential Moving Averages (EMAs)
        fast_ema = data[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[column].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate Signal line (9-day EMA of MACD)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD Histogram
        macd_histogram = macd_line - signal_line

        # Create a DataFrame with calculated values for this security
        security_macd_data = pd.DataFrame({
            f'{column}_MACD': macd_line,
            f'{column}_Signal': signal_line,
            f'{column}_Histogram': macd_histogram
        })

        # Concatenate the security-specific DataFrame to the result DataFrame
        macd_data = pd.concat([macd_data, security_macd_data], axis=1)

    return macd_data

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data):
    data['20_SMA'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['50_SMA'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['TP'] = (data['Close'] + data['Low'] + data['High']) / 3
    data['std'] = data['TP'].rolling(20).std(ddof=0)
    data['MA_TP'] = data['TP'].rolling(20).mean()
    data['BOLU'] = data['MA_TP'] + 2 * data['std']
    data['BOLD'] = data['MA_TP'] - 2 * data['std']
    return data[['20_SMA', '50_SMA', 'TP', 'std', 'MA_TP', 'BOLU', 'BOLD']]

# Function to calculate Supertrend indicator
def calculate_supertrend(data, atr_period, multiplier):
    high = data['High']
    low = data['Low']
    close = data['Close']
    price_diffs = [high - low, high - close.shift(), close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    supertrend = [True] * len(data)

    for i in range(1, len(data.index)):
        curr, prev = i, i - 1
        if close[curr] > final_upperband.iloc[prev]:  # Replace .loc with .iloc
            supertrend[curr] = True
        elif close[curr] < final_lowerband.iloc[prev]:  # Replace .loc with .iloc
            supertrend[curr] = False
        else:
            supertrend[curr] = supertrend[prev]
            if supertrend[curr] == True and final_lowerband.iloc[curr] < final_lowerband.iloc[prev]:  # Replace .loc with .iloc
                final_lowerband.iloc[curr] = final_lowerband.iloc[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband.iloc[curr] = np.nan  # Replace .loc with .iloc

    return pd.DataFrame({'Supertrend': supertrend, 'Final Lowerband': final_lowerband, 'Final Upperband': final_upperband}, index=data.index)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_nifty_50_symbols')
def get_nifty_50_symbols():
    try:
        # Read the Excel file to get the list of Nifty 50 stock symbols
        df = pd.read_excel(excel_file_path, engine='openpyxl')

        # Assuming the Excel file has a column named 'Symbol' that contains the stock symbols
        stock_symbols = df['Symbol'].tolist()

        # Return the list of stock symbols as a JSON response
        return jsonify(stock_symbols)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/analyze', methods=['GET'])
def analyze():
    global excel_file_path
    try:
        stocks = request.args.getlist('stocks')  # Get the selected stocks
        days = request.args.get('days')      # Get the number of days

        # Check if either 'stocks' or 'days' is missing or empty
        if not stocks or not days:
            return "Both 'stocks' and 'days' query parameters are required.", 400

        # Ensure 'days' is a positive integer
        try:
            days = int(days)
            if days <= 0:
                return "'days' must be a positive integer.", 400
        except ValueError:
            return "'days' must be a positive integer.", 400

        df_symbols = pd.read_excel(excel_file_path)

        # Create an empty list to store dataframes for each stock symbol
        dataframes = []

        # Loop through each stock symbol
        for symbol in df_symbols['Symbol']:
            # Fetch historical data for 365 days from Yahoo Finance
            df = download_yahoo_data(symbol, period="365d")
            #print(df)

            if df is not None:
                # Calculate MACD for the downloaded data
                df_macd = get_macd(df)

                # Calculate Bollinger Bands for the downloaded data
                df_bollinger = calculate_bollinger_bands(df)

                # Calculate Supertrend for the downloaded data
                atr_period = 10
                atr_multiplier = 3.0
                supertrend_data = calculate_supertrend(df, atr_period, atr_multiplier)

                # Combine the dataframes
                df = pd.concat([df, df_macd, supertrend_data], axis=1)

                # Save the processed data to an Excel file in the "static" folder
                output_folder = os.path.join("static", "stock_data")
                os.makedirs(output_folder, exist_ok=True)

                stock_file_name = f"{symbol}_processed_data.xlsx"
                stock_file_path = os.path.join(output_folder, stock_file_name)
                df.to_excel(stock_file_path, index=False)

                # Append the dataframe to the list
                dataframes.append(df)

        # You can save the combined data to a single CSV file if needed
        # combined_df = pd.concat(dataframes, ignore_index=True)
        # combined_df.to_csv('processed_data.csv', index=False)

        # Return the MACD DataFrame as HTML
        return render_template('analyze.html')
    except Exception as e:
        return str(e)

@app.route('/load_macd_data', methods=['GET'])


def load_macd_data():
    try:
        df_symbols = pd.read_excel(excel_file_path, engine='openpyxl')
        macd_data_list = []  # Create an empty list to store MACD data

        for symbol in df_symbols['Symbol']:
            # Load MACD data for each symbol based on the symbol's data source
            output_folder = os.path.join("static", "stock_data")
            stock_file_name = f"{symbol}_processed_data.xlsx"
            stock_file_path = os.path.join(output_folder, stock_file_name)
            
            # Specify the delimiter if it's not a comma
            delimiter = ','  # Modify if needed (e.g., '\t' for tab-delimited)
            
            # Read the CSV file with the specified delimiter and encoding
            df_macd = pd.read_excel(stock_file_path, engine='openpyxl')

            # Replace NaN values with null (compatible with JSON)
            df_macd = df_macd.where(pd.notna(df_macd), None)

            if df_macd is not None:
                macd_columns = ['Open_MACD','Open_Signal','Open_Histogram','High_MACD','High_Signal','High_Histogram','Low_MACD','Low_Signal','Low_Histogram','Close_MACD','Close_Signal','Close_Histogram','Adj Close_MACD','Adj Close_Signal','Adj Close_Histogram','Volume_MACD','Volume_Signal','Volume_Histogram']  
                df_macd = df_macd[macd_columns]
                print(df_macd)

        # Return the data as a successful response
        return jsonify({'data': macd_data_list})

    except Exception as e:
        # Return an error response with a message
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
