from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
# Function to create a MongoDB client
def get_client():
    try:
        client = MongoClient(MONGO_URI)        
        print("MongoDB connection successful.")        
        return client
    except ConnectionFailure:
        print("MongoDB connection failed.")
        
        return None

# Create a global client to be reused
client = get_client()

# You can also create a function to get the database object
def get_database(db_name):
    if client is not None:
        return client[db_name]
    else:
        return None

def main():
    # Test the connection
    client = get_client()

if __name__ == "__main__":
    main()