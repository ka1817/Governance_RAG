from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_ingestion import DataIngestion
class Preprocessing:
    def __init__(self,docs):
        self.docs=docs 
    def split(self):
        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=splitter.split_documents(self.docs)
        print(len(chunks)) 
        return chunks
