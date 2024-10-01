from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere

load_dotenv()

llm_cohere = ChatCohere(model="command-r-plus")
embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vectordb_filepath = "faiss_index"

def create_vector_db():
    file_path = "codebasics_faqs.csv"
    loader = CSVLoader(file_path=file_path, source_column="prompt", encoding='cp1252')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_filepath)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_filepath, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    prompt_template = (
        """ Given the following context and a question, generate an answer based in this context only. In the answer try 
        to provide as much text as possible from "response" section in the source document context without making up words. 
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}
        QUESTION: {question}"""
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm_cohere,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

if __name__ == "__main__":
    qa_chain = get_qa_chain()
    print(qa_chain.invoke("Do you provide internship? Do you have EMI option?"))