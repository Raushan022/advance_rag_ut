from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

# combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpfu answer using only the information from these documents. If you can't find the answer in these documents, say "I don't have enough information to answer that question based on the provided documents"
"""

# create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

messages = [
   SystemMessage(content="You are a helpful assistent."),
   HumanMessage(content=combined_input),
]

# invoke the model with combined input
result = model.invoke(messages)

# display the full result and content only
print("\n---- Generated Response ----")
print("content only")
print(result.content)

