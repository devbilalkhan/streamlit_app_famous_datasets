from .client import get_database
from config import DATABASE_NAME
import pandas as pd

def insert_documents(df, collection_name):
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]
    # convert dataset to record
    records = df.to_dict('records')

    result = collection.insert_many(records)
    print(result)
    print(f"Inserting into collection: {collection_name}")
    print(f"Records to insert: {records[-1]}")
    return result.inserted_ids

def load_data_from_mongodb(collection_name, problem_type):
 
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]
    # Convert the collection to a list of dictionaries (each document becomes a dictionary)
    
    data_list = list(collection.find({"ProblemType": problem_type}))
    # Convert the list of dictionaries to a DataFrame
    data_frame = pd.DataFrame(data_list)
    
    # You may want to drop the '_id' column generated by MongoDB
    if '_id' in data_frame.columns:
        data_frame.drop('_id', axis=1, inplace=True)
    
    return data_frame