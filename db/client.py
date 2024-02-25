from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

# Get database username and password from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Escape the username and password using quote_plus
escaped_username = quote_plus(DB_USERNAME)
escaped_password = quote_plus(DB_PASSWORD)

# Construct the MongoDB URI
MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@steamlit-app.sixwgrb.mongodb.net/?retryWrites=true&w=majority&appName=steamlit-app"

#MONGO_URI = "localhost:27017"
# Initialize the MongoDB client
client = None

def get_client():
    """
    Get the MongoDB client, creating it if it doesn't exist.

    Returns:
    pymongo.MongoClient: The MongoDB client.
    """
    global client
    if client is None:
        try:
            client = MongoClient(MONGO_URI)
            print("MongoDB connection successful.")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
    return client

def get_database(db_name):
    """
    Get a database from the MongoDB client.

    Parameters:
    db_name (str): The name of the database to get.

    Returns:
    pymongo.database.Database: The database.
    """
    client = get_client()
    if client is not None:
        return client[db_name]
    else:
        print("No MongoDB client available.")
        return None
