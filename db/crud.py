from .client import get_database
from config import DATABASE_NAME

def insert_documents(df, collection_name):
    """
    Insert multiple documents into a collection.

    :param documents: List of dictionaries, where each dictionary represents a document.
    :param collection_name: The name of the collection to insert the documents into.
    :return: List of the inserted IDs.
    """
    db = get_database(DATABASE_NAME)
    collection = db[collection_name]
    # convert dataset to record
    records = df.to_dict('records')

    result = collection.insert_many(records)
    print(result)
    print(f"Inserting into collection: {collection_name}")
    print(f"Records to insert: {records}")
    return result.inserted_ids