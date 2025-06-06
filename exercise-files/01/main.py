from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from dotenv import load_dotenv

load_dotenv()
model = OpenAI()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)

query = "where did harrison work?"
# docs = vectorstore.similarity_search(query, top_k=1)
# print(docs[0].page_content)

#querying as retriever
# retriever = vectorstore.as_retriever()
# docs = retriever.invoke(query,top_k=1)
# print(docs[0].page_content)

retriever = vectorstore.as_retriever()
retrieval_chain = (
    {
        "context": retriever, "question": RunnablePassthrough(),
    } 
    | prompt 
    | model
    | StrOutputParser()
)

response = retrieval_chain.invoke(query)
print(response)
