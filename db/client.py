from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import quote_plus
import toml

def load_secrets():
    with open(".streamlit/secrets.toml", "r") as toml_file:
        return toml.load(toml_file)

secrets = load_secrets()
mongo = secrets.get("mongo", {})

# Load environment variables from .env file
load_dotenv()

MONGO_URI = mongo.get("MONGO_URI")

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
