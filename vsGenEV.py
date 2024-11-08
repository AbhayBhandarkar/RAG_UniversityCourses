import pymongo
import json
from sentence_transformers import SentenceTransformer

# MongoDB connection
conn = pymongo.MongoClient("mongodb+srv://abhaybhandarkar2:<db_password>@cluster0.jzzsl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = conn['education']
coll = db['courses']

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to the JSON file
file_path = "/Users/mangalabhandarkar/Downloads/Archive (3)/medical_engineering_courses.json"

# Open JSON file and read line by line
with open(file_path, "r") as file:
    documents = json.load(file) 

for doc in documents:
    if 'fullplot' in doc:
        doc['plot_embedding_hf'] = model.encode(doc['fullplot'], normalize_embeddings=True).tolist()

        coll.insert_one(
            doc                     # Insert if document does not exist
        )

print("Embeddings have been added/updated for documents in MongoDB.")