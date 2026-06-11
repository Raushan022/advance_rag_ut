import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
   """Load all text files from the docs directory"""
   print(f"Loading documents from {docs_path}...")

   if not os.path.exists(docs_path):
      raise FileNotFoundError(f"The directory {docs_path} does not exists")
   
   # load all .txt files from docs directory
   loader = DirectoryLoader(
      path=docs_path,
      glob="*.txt",
      loader_cls=TextLoader,
      loader_kwargs={"encoding": "utf-8"}
   )

   documents = loader.load()

   if len(documents) == 0:
      raise FileNotFoundError(f"No .txt files found in {docs_path} does not exists. Please add you company documents")
   
   # for i, doc in enumerate(documents[:2]): #show first two document
   #    print(f"\nDocument {i+1}:")
   #    print(f"Source: {doc.metadata["source"]}")
   #    print(f"Content length: {len(doc.page_content)} characters")
   #    print(f"content preview: {doc.page_content[:100]}...")
   #    print(f"metadata: {doc.metadata}")

   return documents

def split_documents(documets, chunk_size=1000, chunk_overlap=200):
   """split documents into smaller chunks with overlap"""
   print("splitting documents into chunks")

   text_splitter = CharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
   )

   chunks = text_splitter.split_documents(documets)

   # if chunks:

   #    for i, chunk in enumerate(chunks[:5]): #show first two document
   #       print(f"\n--- Chunk {i+1} ----:")
   #       print(f"Source: {chunk.metadata["source"]}")
   #       print(f"length: {len(chunk.page_content)} characters")
   #       print(f"content: ")
   #       print(chunk.page_content)
   #       print("-" * 50)

   #    if len(chunks) > 5:
   #       print(f"\n.. and {len(chunks) - 5} more chunks")

   return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
   """create and persist ChromaDB vector store"""
   print("creating embedding and storing in ChromaDB")

   embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

   print("----creating vector store----")
   vectorstore = Chroma.from_documents(
      documents=chunks,
      embedding=embedding_model,
      persist_directory=persist_directory,
      collection_metadata={"hnsw:space": "cosine"}
   )

   print("----Finished creating vector store----")

   print(f"vector store created and saved to {persist_directory}")
   return vectorstore


def main():
   print("Main function")

   # 1. Loading the file
   documents = load_documents(docs_path="docs")

   # 2. Chunking the files
   chunks = split_documents(documents)

   # 3. Embedding and storing in vector DB
   vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
   main()
