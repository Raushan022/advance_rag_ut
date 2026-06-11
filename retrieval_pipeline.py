from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db\chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
   persist_directory=persistent_directory,
   embedding_function=embedding_model,
   collection_metadata={"hnsw:space": "cosine"}
)

# search for relevant documents
query = "In what year did Tesla begin production of Roadster?"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
# display results
print("----context----")
for i, doc in enumerate(relevant_docs, 1):
   print(f"Documetn {i}:\n{doc.page_content}\n")