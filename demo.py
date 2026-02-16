from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

# 1️⃣ Load document
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings (USE embedding model, NOT chat model)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4️⃣ Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6️⃣ Load LLM
llm = OllamaLLM(model="llama3.1")

# 7️⃣ Query
query = "What is generative ai?"

retrieved_docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

# 8️⃣ Generate answer
response = llm.invoke(prompt)

print("\nAnswer:\n", response)