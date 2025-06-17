from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()
ATLAS_CONNECTION_STRING = os.getenv("ATLAS_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

# Connect to your Atlas cluster
client = MongoClient("mongodb+srv://ernestoorozconlp:pClHVqey84A9DKAd@clusterlangchain.es7v444.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLangChain")

# Define collection and index name
collection = client[DB_NAME][COLLECTION_NAME]

# if client:
    # print("Connected to MongoDB Atlas")
    # print(client.list_database_names())

# Load the sample data (PDF document)
loader = PyPDFLoader("imitation_of_the_christ.pdf")
data = loader.load()

# Split PDF into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(data)

# Instantiate the vector store
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=[],
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection=collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)
# Instantiate Atlas Vector Search as a retriever
retriever = vector_store.as_retriever(
   search_type = "similarity",
   search_kwargs = { "k": 1 }
)

# Define a prompt template
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)
 

# Construct a chain to answer questions on your data

reg_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | model
    | StrOutputParser()
)

