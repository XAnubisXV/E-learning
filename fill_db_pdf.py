import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
pdf_path = "ZOGG.pdf"

# PDF laden
reader = PdfReader(pdf_path)
pages = []

#for page in reader.pages:
#    text = page.extract_text()
#    if text:
#        pages.append(text)

pages = []
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        pages.append({
            "page": page_number,
            "text": text
        })

#full_text = "\n".join(pages)
print(f"Loaded PDF with {len(pages)} pages")

# Text in kleinere Bestandteile (Chunks) aufteilen
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

#chunks = text_splitter.split_text(full_text)
#print(f"Created {len(chunks)} chunks")
# Vektordatenbank einrichten und Kollektion erstellen
client = chromadb.PersistentClient(path="./chroma_db")
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jinaai/jina-embeddings-v2-base-de"
)
collection = client.get_or_create_collection(
    "verfahrenstechnik",
    embedding_function=emb
)
# PDF-Bestandteile (Chunks) zur Vektordatenbank hinzufügen
#ids = [str(uuid4()) for _ in chunks]
#metadatas = [{"source": pdf_path, "page": i} for i in range(len(chunks))]

#collection.add(
#    documents=chunks,
#    metadatas=metadatas,
#    ids=ids
#)
ids = []
metadatas = []
chunks = []
for page in pages:
    page_chunks = text_splitter.split_text(page["text"])

    for chunk in page_chunks:
        chunks.append(chunk)
        metadatas.append({
            "source": pdf_path,
            "page": page["page"]
        })
        ids.append(str(uuid4()))

collection.add(
    documents=chunks,
    metadatas=metadatas,
    ids=ids
)

print("Stored PDF in Chroma.")

# Die Datenbank abfragen
query = "Welche anderen Filtrationsmethoden gibt es außer Druckfiltration?"
results = collection.query(
    query_texts=[query],
    n_results=5,
)

print(results)
