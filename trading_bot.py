import ccxt
import json
import os
from dotenv import load_dotenv
import pandas as pd
import warnings
import numpy as np
from ta import add_all_ta_features
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import datetime
import time
import requests

load_dotenv()
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

min_liquidity=10000 # Min amount of volume in USD
MIN_DATA_POINTS = 25
MIN_ITERATIONS_BETWEEN_BUYS = 0  # Set the minimum number of iterations between buy orders (bot runs every 2 hours so it will buy every 2 hours)
buy_counter = MIN_ITERATIONS_BETWEEN_BUYS
buy_quantity = 10

exchange = ccxt.cryptocom({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

def log(message, log_file="trading_bot.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(log_file, "a") as f:
        f.write(log_message + "\n")

def usd_usdt_pair_exists():
    markets = exchange.load_markets()
    return 'USDT/USD' in markets

usd_usdt_exists = usd_usdt_pair_exists()
log(f"USDT/USD trading pair exists: {usd_usdt_exists}")

def get_preferred_quote_currency(base_currency, min_liquidity):
    quote_currencies = ['USD', 'USDT']
    max_volume = -1
    preferred_quote = None

    for quote_currency in quote_currencies:
        symbol = f"{base_currency}/{quote_currency}"
        markets = exchange.load_markets()
        if symbol not in markets:
            continue

        try:
            now = exchange.milliseconds()
            one_day_ago = now - 24 * 60 * 60 * 1000
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=one_day_ago)
            volume_24h = sum([candle[5] for candle in ohlcv])

            if volume_24h >= min_liquidity and volume_24h > max_volume:
                max_volume = volume_24h
                preferred_quote = quote_currency
        except (ccxt.BadSymbol, ccxt.NetworkError, ccxt.ExchangeError):
            continue

    return preferred_quote

def get_trading_pairs(min_liquidity):
    markets = exchange.load_markets()
    trading_pairs = set()  # Use a set to store unique trading pairs

    for symbol, market in markets.items():
        base_currency = market['base']
        preferred_quote_currency = get_preferred_quote_currency(base_currency, min_liquidity)

        if preferred_quote_currency:
            trading_pairs.add(f"{base_currency}/{preferred_quote_currency}")

    return list(trading_pairs)  # Convert the set back to a list

trading_pairs = get_trading_pairs(min_liquidity)
log(f"Selected trading pairs: {trading_pairs}")

def get_positions(trading_pairs):
    balance = exchange.fetch_balance()
    positions = {}

    for symbol in trading_pairs:
        base_currency = symbol.split('/')[0]  # It should be the base currency, not the quote currency.
        if base_currency in balance['free'] and balance['free'][base_currency] > 0:
            positions[symbol] = 'asset'
        else:
            positions[symbol] = 'cash'

    return positions

positions = get_positions(trading_pairs)

## Get Historical Data to train the Machine Learning module ##
warnings.filterwarnings("ignore", category=RuntimeWarning)

def fetch_historical_data(symbol, timeframe='1h', limit=3000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_technical_indicators(df):
    df = add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)
    return df

# Create a dictionary to store historical data and technical indicators for all trading pairs
historical_data_dict = {}

for symbol in trading_pairs:
    success = False
    limit = 3000
    while not success:
        try:
            historical_data = fetch_historical_data(symbol, limit=limit)
            if len(historical_data) < MIN_DATA_POINTS:  # Set a minimum threshold for the number of data points
                log(f"Skipping {symbol} due to insufficient data ({len(historical_data)} data points)")
                break
            #log(f"Before indicators for {symbol} with {len(historical_data)} data points.")
            df = calculate_technical_indicators(historical_data)
            #log(f"After indicators for {symbol} with {len(df)} data points.")
            log(f"{symbol} has {len(df)} data points.")
            historical_data_dict[symbol] = df
            success = True
        except ValueError as e:
            log(f"Error for {symbol} with {len(historical_data)} data points: {e}. Retrying with a larger limit.")
            limit += 1000
        except IndexError as e:
            log(f"IndexError for {symbol} with {len(historical_data)} data points: {e}")
            break

# Print the first few rows of the DataFrame for the first trading pair
log(historical_data_dict[trading_pairs[0]].head())

## Training the Machine Learning Module with that data ##
# Prepare target variable based on moving average crossover strategy
#def create_target(df, short_window=3, long_window=6): # 3 = 36 hours, 6 = 72 hours
#def create_target(df, short_window=12, long_window=24): # 12 = 24 hours, 24 = 48 hours
def create_target(df, short_window=6, long_window=12): # 6 = 72 hours, 12 = 144 hours
    df['short_mavg'] = df['close'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window).mean()
    df['target'] = np.where(df['short_mavg'] > df['long_mavg'], 1, -1)
    df = df.dropna()
    return df

# Train a separate model for each trading pair
models = {}
tscv = TimeSeriesSplit(n_splits=5)  # Use 5-fold cross-validation

for symbol, df in historical_data_dict.items():
    if symbol == "USDT/USD":
        log(f"Skipping {symbol} as it is used only for exchanging money")
        continue

    if len(df) < MIN_DATA_POINTS:
        log(f"Skipping {symbol} due to insufficient data ({len(df)} data points)")
        continue

    # Create target variable
    df = create_target(df)

    # Log the number of samples after creating the target variable
    #log(f"Number of samples for {symbol} after creating target variable: {len(df)}")

    # Skip the trading pair if the number of samples is less than or equal to the number of splits
    if len(df) <= tscv.n_splits:
        log(f"Skipping {symbol} due to insufficient samples after creating target variable ({len(df)} samples)")
        continue

    # Define target_col, input_cols, and other parameters based on your strategy
    target_col = 'target'
    excluded_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'short_mavg', 'long_mavg', 'target']
    input_cols = [col for col in df.columns if col not in excluded_cols]

    # Prepare the data
    X = df[input_cols]
    y = df[target_col]

    # Train the Random Forest model using cross-validation
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=tscv, n_jobs=-1)

    # Fit the model on the entire dataset
    model.fit(X, y)

    # Evaluate the model's performance
    mean_cv_score = cv_scores.mean()
    log(f"Mean cross-validation score for {symbol} with n_estimators=200: {mean_cv_score:.2f}")

    # Store the trained model in the models dictionary
    models[symbol] = model

    #if isinstance(X, pd.DataFrame):
        #print("Missing values in X:", X.isna().any().any())
    #else:
        #print("Missing values in X:", np.isnan(X).any())

    #print("Missing values in y:", np.isnan(y).any())

## Executing the Trades ##
recent_trades = {}
def execute_trade(symbol, position, predicted_trend, trading_fee=0.00075, allow_buy=True):
    # Fetch the trading pair's market information
    market_info = exchange.market(symbol)

    # Retrieve the minimum trade amount and quantity tick size from the market information
    min_trade_amount = market_info['limits']['amount']['min']
    quantity_tick_size = market_info['precision']['amount']

    # Calculate the amount to trade, factoring in the trading fee
    trade_amount = min_trade_amount / (1 + trading_fee)

    # Adjust trade amount to be a multiple of the quantity tick size
    trade_amount = quantity_tick_size * round(trade_amount / quantity_tick_size) * buy_quantity
    if trade_amount < min_trade_amount:
        log(f"Adjusted trade amount {trade_amount} is less than minimum trade amount {min_trade_amount}")
        return 'cash', False  # Return both the position and the transfer_made flag as False

    # Fetch the available balance
    balance = exchange.fetch_balance()
    base_currency, quote_currency = symbol.split('/')
    log(f"Fetching available balance for {base_currency} and {quote_currency}...")

    has_base_currency = base_currency in balance['free']
    if not has_base_currency and predicted_trend == 1:
        position = 'cash'  # Set the position to 'cash' when the currency (asset) is not found and the predicted_trend is 1
    elif not has_base_currency:
        return 'cash', False  # Return both the position and the transfer_made flag as False
    
    # Check if the base currency exists in the balance
    if base_currency in balance['free']:
        base_currency_balance = balance['free'][base_currency]
    else:
        log(f"Currency {base_currency} not found in balance.")
        base_currency_balance = 0

    # Get the preferred quote currency based on liquidity
    preferred_quote_currency = get_preferred_quote_currency(base_currency, min_liquidity)

    # If the preferred quote currency is different from the quote currency in the symbol, update the symbol
    if preferred_quote_currency and preferred_quote_currency != quote_currency:
        quote_currency = preferred_quote_currency
        symbol = f"{base_currency}/{quote_currency}"

    if base_currency not in balance['free']:
        if predicted_trend == 1:
            position = 'cash'  # Set the position to 'cash' when the currency (asset) is not found and the predicted_trend is 1
        else:
            return 'cash', False  # Return both the position and the transfer_made flag as False

    # Trade between USD and USDT if necessary
    transfer_made = False
    if usd_usdt_exists and (base_currency == "USD" or base_currency == "USDT"):
        target_currency = "USD" if base_currency == "USDT" else "USDT"
        available_balance = balance['free'][base_currency]

        if available_balance > 0:
            # Calculate the required amount for the buy order
            required_amount = 0
            if position == "cash" and predicted_trend == 1 and allow_buy:
                required_amount = balance['free'][quote_currency] * (buy_quantity + trading_fee)  # Calculate the required amount for the buy order

            # Calculate the amount to transfer between USD and USDT
            transfer_amount = min(available_balance, required_amount + trading_fee * required_amount)
            if transfer_amount > 0:
                log(f"Trading {base_currency} to {target_currency} with amount {transfer_amount}")
                order = exchange.create_market_sell_order(f"{base_currency}/{target_currency}", transfer_amount)
                order_info = exchange.fetch_order(order['id'], f"{base_currency}/{target_currency}")  # Fetch the complete order information
                # log(f"Order details: {order_info}")
                transfer_made = True
                position = 'cash'  # Update the position after trading USDT/USD or USD/USDT

    if predicted_trend == 1 and position == 'cash' and allow_buy:
        # Fetch the current ticker price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['ask']
        log(f"Current price for {symbol}: {current_price}")

        # Check if the available balance is sufficient to place the buy order
        required_balance = trade_amount * current_price * buy_quantity * (1 + trading_fee)
        available_balance = balance['free'][quote_currency]  # Check the balance of quote_currency instead of base_currency
        log(f"Required balance for buy order: {required_balance}. Available balance: {available_balance}")

        if available_balance >= required_balance:
            # Buy
            trade_amount_with_fee = max(min_trade_amount, trade_amount)
            # Adjust trade amount with fee to be a multiple of the quantity tick size
            trade_amount_with_fee = quantity_tick_size * round(trade_amount_with_fee / quantity_tick_size)
            log(f"Buying {symbol} with amount {trade_amount_with_fee}")
            log(f"Attempting to place buy order for {symbol}...")
            order = exchange.create_market_buy_order(symbol, trade_amount_with_fee)
            order_info = exchange.fetch_order(order['id'], symbol)  # Fetch the complete order information
            # log(f"Order details: {order_info}")
            position = 'asset'
            
            # Prepare the trade details
            trade_details = {
                'symbol': symbol,
                'amount': trade_amount_with_fee,
                'trade_type': 'BUY',
                'balance': balance['free'][quote_currency],
                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'order_id': order['id'],
                'current_price': current_price
            }

            # Send the trade details to the external server
            try:
                response = requests.post('http://192.168.1.90:5006/trades', json=trade_details)
                if response.status_code == 200:
                    log(f'Successfully posted trade details: {trade_details}')
                else:
                    log(f'Error posting trade details: {trade_details}. Response: {response.text}')
            except requests.exceptions.RequestException as err:
                log(f'Error posting trade details: {trade_details}. Error: {err}')
            else:
                log(f"Insufficient balance to buy {symbol} (Required: {required_balance}, Available: {available_balance})")

    elif predicted_trend == -1 and has_base_currency and balance['free'][base_currency] >= min_trade_amount:
        # Sell all if available balance is more than minimum trade amount
        trade_amount = base_currency_balance  # Set trade amount to all available balance

        # Fetch the current ticker price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['ask']
        log(f"Current price for {symbol}: {current_price}")

        # Fetch the last purchase price from the endpoint
        response = requests.get(f"http://localhost:5006/last_purchase_price?symbol={symbol}")
        if response.status_code == 200:
            data = response.json()
            last_purchase_price = data.get('last_purchase_price', 0)

            # Compare the current price with the last purchase price
            if current_price < last_purchase_price:
                log(f"Current price is less than the purchase price. Skipping sell for {symbol}")
                return position, transfer_made
        else:
            log(f"Could not fetch the last purchase price for {symbol}. Skipping sell.")
            return position, transfer_made
        
        log(f"Selling {symbol} with amount {trade_amount}")
        order = exchange.create_market_sell_order(symbol, trade_amount)
        order_info = exchange.fetch_order(order['id'], symbol)  # Fetch the complete order information
        # log(f"Order details: {order_info}")
        position = 'cash'
        # Prepare the trade details
        trade_details = {
            'symbol': symbol,
            'amount': trade_amount,
            'trade_type': 'SELL',
            'balance': balance['free'][quote_currency],
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'order_id': order['id'],
            'current_price': current_price
        }
        # Send the trade details to the external server
        try:
            response = requests.post('http://192.168.1.90:5006/trades', json=trade_details)
            if response.status_code == 200:
                log(f'Successfully posted trade details: {trade_details}')
            else:
                log(f'Error posting trade details: {trade_details}. Response: {response.text}')
        except requests.exceptions.RequestException as err:
            log(f'Error posting trade details: {trade_details}. Error: {err}')
    else:
        log(f"No trade for {symbol}: position={position}, predicted_trend={predicted_trend}, allow_buy={allow_buy}, balance={balance['free'][base_currency] if has_base_currency else 0}")
    return position, transfer_made

def fetch_total_balance():
    # Fetch the total balance
    balance = exchange.fetch_balance()
    total_balance = balance['total']  # 'total' key contains the total balance for each currency

    # Fetch all the prices
    prices = exchange.fetch_tickers()

    total_usd_balance = 0.0

    for currency, amount in total_balance.items():
        if currency == 'USD':
            total_usd_balance += amount
        elif currency + '/USD' in prices:  # If the pair exists
            # Convert the amount to USD and add it to the total balance
            usd_price = prices[currency + '/USD']['last']  # Fetch the last price of the currency
            total_usd_balance += amount * usd_price

    balance_data = {'total_usd_balance': total_usd_balance}
    try:
        response = requests.post('http://192.168.1.90:5006/balance', json=balance_data)
        if response.status_code == 200:
            print('Successfully posted balance data:', balance_data)
        else:
            print('Error posting balance data:', response.text)
    except requests.exceptions.RequestException as err:
        print('Error posting balance data:', err)

buy_counters = {symbol: 0 for symbol in trading_pairs}
sent_pairs = set()
next_iteration_time = None
while True:
    try:
        log(f"Start of iteration: Buy counter: {buy_counter}")

        # Fetch and send total balance
        fetch_total_balance()
        
        for symbol in trading_pairs:
            # Check if a model exists for the trading pair
            if symbol not in models:
                log(f"Skipping {symbol} due to missing model")
                continue

            # Fetch latest data and calculate technical indicators
            latest_data = fetch_historical_data(symbol, limit=100)

            if len(latest_data) < MIN_DATA_POINTS:
                log(f"Skipping {symbol} due to insufficient recent data ({len(latest_data)} data points)")
                continue

            latest_data = calculate_technical_indicators(latest_data)

            # Make predictions using the trained model for the specific trading pair
            X_latest = pd.DataFrame([latest_data[input_cols].iloc[-1]], columns=input_cols)
            predicted_trend = models[symbol].predict(X_latest)[0]
            predicted_probabilities = models[symbol].predict_proba(X_latest)[0]
            log(f"Predicted probabilities for {symbol}: {predicted_probabilities}")

            # Execute trades based on the trading logic
            allow_buy = buy_counter >= MIN_ITERATIONS_BETWEEN_BUYS
            positions[symbol], transfer_made = execute_trade(symbol, positions[symbol], predicted_trend, allow_buy=allow_buy)

            # Send trading pair to the Flask application
            if symbol not in sent_pairs:
                try:
                    response = requests.post('http://192.168.1.90:5006/pairs', json={'pair': symbol})
                    if response.status_code == 200:
                        log(f'Successfully posted pair: {symbol}')
                        sent_pairs.add(symbol)
                    else:
                        log(f'Error posting pair: {symbol}. Response: {response.text}')
                except requests.exceptions.RequestException as err:
                    log(f'Error posting pair: {symbol}. Error: {err}')

            # Retry the trade if a transfer was made
            if transfer_made:
                positions[symbol], _ = execute_trade(symbol, positions[symbol], predicted_trend, allow_buy=allow_buy)

        # Increment the buy counter and reset it if it reaches the threshold
        buy_counter = (buy_counter + 1) % (MIN_ITERATIONS_BETWEEN_BUYS + 1)

        # Wait for a specified interval before the next iteration
        next_iteration_time = datetime.datetime.now() + datetime.timedelta(hours=1)

        # Prepare the time details
        time_details = {'next_iteration_time': next_iteration_time.strftime('%Y-%m-%d %H:%M:%S')}

        # Send the time details to the external server
        try:
            response = requests.post('http://192.168.1.90:5006/time', json=time_details)
            if response.status_code == 200:
                log(f'Successfully posted time details: {time_details}')
            else:
                log(f'Error posting time details: {time_details}. Response: {response.text}')
        except requests.exceptions.RequestException as err:
            log(f'Error posting time details: {time_details}. Error: {err}')
        time.sleep(1 * 60 * 60)  # Example: wait for 1 hours
    except Exception as e:
        log(f"Error: {e}")
        time.sleep(60 * 5)  # Wait for 5 minutes before retrying
