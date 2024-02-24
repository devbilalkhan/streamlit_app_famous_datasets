from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')

# Function to create a MongoDB client
def get_client():
    try:
        client = MongoClient(MONGO_URI)
        print("MongoDB connection successful.")
        return client
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return None

# Create a global client to be reused
client = get_client()

# Function to get the database object
def get_database(db_name):
    if client is not None:
        return client[db_name]
    else:
        print("No MongoDB client available.")
        return None

