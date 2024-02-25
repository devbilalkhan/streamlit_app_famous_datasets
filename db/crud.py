from .client import get_database
from config import DATABASE_NAME, DATASET_COLLECTION_NAME
import pandas as pd

def insert_documents(df, collection_name):
    """
    Inserts documents into a MongoDB collection.

    Parameters:
    df (pandas.DataFrame): The DataFrame to insert.
    collection_name (str): The name of the collection to insert into.

    Returns:
    list: The IDs of the inserted documents.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]
    records = df.to_dict('records')
    result = collection.insert_many(records)
    return result.inserted_ids

def load_data_from_mongodb(collection_name, problem_type=None):
    """
    Loads data from a MongoDB collection.

    Parameters:
    collection_name (str): The name of the collection to load from.
    problem_type (str): The problem type to filter by.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]
    data_list = list(collection.find({"ProblemType": problem_type}))
    data_frame = pd.DataFrame(data_list)
    if '_id' in data_frame.columns:
        data_frame.drop('_id', axis=1, inplace=True)
    return data_frame


def create_records(data, collection):
        """
        Inserts a new document or documents into the collection.
        :param data: dict or list of dicts to insert into the collection
        """
        db = get_database(DATABASE_NAME)
        collection = db[collection]

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        if isinstance(data, dict):
            collection.insert_one(data)
            return data
        elif isinstance(data, list):
            collection.insert_many(data)
            return data
        else:
            raise TypeError("Data must be a DataFrame, dictionary, or list of dictionaries.")


def get_dataset_name(collection):
    """
    Checks if datasets exist in the collection, if not, creates them.

    Parameters:
    dataset_list (list): A list of dictionaries, each containing a 'dataset_name' key.
    collection (str): The name of the collection to check and insert into.

    Returns:
    list: A list of distinct dataset names in the collection.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection]

    all_datasets = list(collection.distinct('dataset_name'))
    if all_datasets is None:
        return None
    return all_datasets


def check_and_create_dataset_name(dataset_list, collection):
    """
    Checks if datasets exist in the collection, if not, creates them.

    Parameters:
    dataset_list (list): A list of dictionaries, each containing a 'dataset_name' key.
    collection (str): The name of the collection to check and insert into.

    Returns:
    list: A list of distinct dataset names in the collection.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection]

    for dataset_name in dataset_list:
        if collection.find_one({'dataset_name': dataset_name}) is None:
            result = collection.insert_one({'dataset_name': dataset_name.lower()})
    all_datasets = list(collection.distinct('dataset_name'))
    return all_datasets


def check_and_create_single_ds_name(dataset_name, collection):
    """
    Checks if a dataset exists in the collection, if not, creates it.

    Parameters:
    dataset_name (str): The name of the dataset.
    collection (str): The name of the collection to check and insert into.

    Returns:
    str: The name of the dataset if it exists or was created, None otherwise.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection]

    if collection.find_one({'dataset_name': dataset_name}) is None:
        result = collection.insert_one({'dataset_name': dataset_name.lower()})
        if result.inserted_id:
            return dataset_name

    return None

def delete_dataset_records(dataset_name, collection):
    """
    Deletes all records with a given dataset_name in the collection.

    Parameters:
    dataset_name (str): The name of the dataset to delete.
    collection (str): The name of the collection to delete from.

    Returns:
    int: The count of deleted documents in the collection.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection]
    
    # Delete all documents that have the dataset_name
    result = collection.delete_many({'dataset_name': dataset_name})
    
    # Return the count of deleted documents
    return result.deleted_count

def delete_collection(collection_name):
    """
    Deletes an entire collection from the database.

    Parameters:
    collection_name (str): The name of the collection to delete.

    Returns:
    bool: True if the collection was dropped successfully, False otherwise.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]    
   
    collection.drop()
    print(f"The collection '{collection_name}' was deleted successfully.")
    return True

    # Handle the exception (e.g., collection does not exist)
    print(f"An error occurred while deleting the collection: {e}")
    return False