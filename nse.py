from nsetools import Nse

# Create an instance of Nse
nse = Nse()

# Get the list of Nifty 50 stock symbols
nifty_symbols = nse.get_stock_codes()

# Filter the Nifty 50 stocks (you may need to clean the data further)
nifty_50_symbols = [symbol for symbol in nifty_symbols.values() if symbol.endswith('.NS')]

# Print the list of Nifty 50 stock symbols
for symbol in nifty_50_symbols:
    print(symbol)
