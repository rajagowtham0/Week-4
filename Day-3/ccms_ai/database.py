from pymongo import MongoClient
# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")
# Access Database
db = client["ccms_ai"]  
# Access Collection
collection = db["patientrecording"]   

# fetching all patient cases from mongodb
def fetch_all_cases():
    documents = list(collection.find({}, {"_id": 0}))
    return documents