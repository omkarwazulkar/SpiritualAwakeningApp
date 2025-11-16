# ===== gita_pipeline.py =====
import os
import pandas as pd
from dotenv import load_dotenv
from vectorstore import generateEmbeddings
from semantic_search import retrieveRelevantDocs
from generation import explainSelectedVerses

load_dotenv()

# Configuration
DATA_DIR = "data"
INDEX_NAME = "gita-index"

# Load Processed Gita Dataset
df = pd.read_csv(os.path.join(DATA_DIR, "structured_gita.csv"))

# Main Function
if __name__ == "__main__":
    
    print("🚀 Starting Bhagavad Gita Processing Pipeline...\n")
    # gitaDf = loadAndProcessGita()
    vectorStore = generateEmbeddings(df, INDEX_NAME)

    userQuestion = "What does Krishna advise about controlling your mind?"
    topDocs = retrieveRelevantDocs(userQuestion, vectorStore)
    explainSelectedVerses(topDocs)

    print("\n🎉 Done. Ready For Web Integration.")


    