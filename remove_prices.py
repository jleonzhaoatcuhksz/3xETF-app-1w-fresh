import sqlite3

# Connect to the database
conn = sqlite3.connect('c:/Users/JLZ/codebuddy/Projects/3xETF-app-1w-lean-fresh/etf_data.db')
cursor = conn.cursor()

# Count rows before deletion (for verification)
cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol IN ('TECL', 'TECS');")
count_before = cursor.fetchone()[0]
print(f"Rows before deletion: {count_before}")

# Remove TECL and TECS from prices table
cursor.execute("DELETE FROM prices WHERE symbol IN ('TECL', 'TECS');")
conn.commit()

# Verify deletion
cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol IN ('TECL', 'TECS');")
count_after = cursor.fetchone()[0]
print(f"Rows after deletion: {count_after}")

if count_after == 0:
    print("All TECL and TECS rows have been successfully removed from the prices table.")
else:
    print(f"Warning: {count_after} rows still exist for TECL and TECS.")

# Show sample of remaining TECL/TECS if any exist
if count_after > 0:
    cursor.execute("SELECT * FROM prices WHERE symbol IN ('TECL', 'TECS') LIMIT 5;")
    remaining = cursor.fetchall()
    print("Sample remaining rows:")
    for row in remaining:
        print(row)

# Close the connection
conn.close()