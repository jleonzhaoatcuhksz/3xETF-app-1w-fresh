import sqlite3

# Connect to the database
conn = sqlite3.connect('c:/Users/JLZ/codebuddy/Projects/3xETF-app-1w-lean-fresh/etf_data.db')
cursor = conn.cursor()

# Remove TECL and TECS
cursor.execute("DELETE FROM etfs WHERE symbol IN ('TECL', 'TECS');")
conn.commit()

# Verify deletion
cursor.execute("SELECT * FROM etfs WHERE symbol IN ('TECL', 'TECS');")
rows = cursor.fetchall()
if len(rows) == 0:
    print("TECL and TECS have been successfully removed from the etfs table.")
else:
    print("Warning: Some rows may still exist:")
    for row in rows:
        print(row)

# Close the connection
conn.close()