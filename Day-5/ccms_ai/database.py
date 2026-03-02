from pymongo import MongoClient
from config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME
# client, Database, and MongoDB URL information from config.py
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
# fetching all data
def fetch_all_cases():
    return list(collection.find({}, {"_id": 0}))