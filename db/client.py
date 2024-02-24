from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


MONGO_URI = "mongodb://localhost:27017/"

# Function to create a MongoDB client
def get_client():
    try:
        client = MongoClient(MONGO_URI)        
        
        client.admin.command('ismaster')
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
