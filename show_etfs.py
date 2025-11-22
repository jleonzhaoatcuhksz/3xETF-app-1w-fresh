import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('c:/Users/JLZ/codebuddy/Projects/3xETF-app-1w-lean-fresh/etf_data.db')

# Query the `etfs` table and display as a DataFrame
df = pd.read_sql_query("SELECT * FROM etfs;", conn)
print(df)

# Close the connection
conn.close()