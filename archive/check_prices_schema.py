import sqlite3

# Connect to database
conn = sqlite3.connect('etf_data.db')
cursor = conn.cursor()

# Get the schema of the prices table
cursor.execute("PRAGMA table_info(prices)")
columns = cursor.fetchall()

print("Current prices table schema:")
print("=" * 50)
for column in columns:
    cid, name, data_type, not_null, default, pk = column
    print(f"{cid:2}: {name:<20} {data_type:<10} {'NOT NULL' if not_null else 'NULL':<8} {'PK' if pk else ''}")

conn.close()