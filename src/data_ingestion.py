from langchain_community.document_loaders import PyPDFLoader
class DataIngestion:
    def __init__(self,path):
        self.path=path 
    def ingest(self):
        loader = PyPDFLoader(self.path)
        docs = loader.load()
        print("Number of pages loaded:", len(docs))

        return docs 

