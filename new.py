import pandas as pd

# Replace 'your_excel_file.xlsx' with the path to your Excel file
excel_file_path = 'stock_list.xlsx'

# Replace 'Column_Name' with the name of the column you want to extract
column_name = 'Symbol'


# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)
    
# Extract the specified column as a list
column_data = df[column_name].tolist()
    
# Print the list
print(f"The data in '{column_name}' column as a list:")
print(column_data)

from flask import Flask, render_template

app = Flask(__name__)

# Sample Python list with data for the dropdown
dropdown_data = column_data

@app.route('/')
def index():
    
    return render_template('index.html', nse_stock_symbols=dropdown_data)

if __name__ == '__main__':
    app.run(debug=True)

