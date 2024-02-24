from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
import streamlit as st
from urllib.parse import quote_plus
# Load environment variables from .env file
load_dotenv()


DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# escape the username and password using quote_plus
escaped_username = quote_plus(DB_USERNAME)
escaped_password = quote_plus(DB_PASSWORD)

#MONGO_URI = f"mongodb+srv://{escaped_username}:{escaped_password}@steamlit-app.sixwgrb.mongodb.net/?retryWrites=true&w=majority&appName=steamlit-app"

MONGO_URI = f"mongodb+srv://bilalkhan:79A8UIqFu6BEWiwg@steamlit-app.sixwgrb.mongodb.net/?retryWrites=true&w=majority&appName=steamlit-app"

client = None

# @st.cache_resource
def get_client():
    global client
    if client is None:
        try:
            client = MongoClient(MONGO_URI)
            print("MongoDB connection successful.")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
    return client




def get_database(db_name):
    client = get_client()
    if client is not None:
        return client[db_name]
    else:
        print("No MongoDB client available.")
        return None
