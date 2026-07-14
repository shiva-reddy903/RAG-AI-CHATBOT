import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vector_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

TOP_K = 20