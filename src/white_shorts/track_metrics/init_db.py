#!/usr/bin/env python
import sqlite3, os
DB_PATH = "whiteshorts.db"
with open("schema.sql", "r", encoding="utf-8") as f:
    schema = f.read()
conn = sqlite3.connect(DB_PATH)
conn.executescript(schema)
conn.commit()
conn.close()
print(f"Initialized DB at {os.path.abspath(DB_PATH)}")
