from databricks import sql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_SQL_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

print("🔄 Testing Databricks SQL Connection...")

try:
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_SQL_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    print("✅ Databricks SQL Warehouse Connection Successful!")
except Exception as e:
    print(f"❌ Databricks SQL Connection Failed: {e}")
