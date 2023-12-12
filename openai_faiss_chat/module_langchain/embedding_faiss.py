from dotenv import load_dotenv
import openai
# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


OPENAI_API_KEY = None
MODEL = "text-embedding-ada-002"
try:
    load_dotenv()
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
except:
    pass

def textfile2faiss(filepath:str,openai_api=OPENAI_API_KEY,resultpath=None)->str: 
    if resultpath==None:
        resultpath=os.path.dirname(filepath)
    if os.path.exists(resultpath):
        pass
    else:
        return
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    
    basename = os.path.basename(filepath)
    filenames=os.path.splitext(basename)
    ext=filenames[1]
    filename=filenames[0]
    if ext.upper()==".PDF":
        loader = PyPDFLoader(filepath)
    elif ext.upper()==".DOCX":
        loader = Docx2txtLoader(filepath)
    else:
        return
    document = loader.load()
    text = text_splitter.split_documents(document)
    faissdb = FAISS.from_documents(text, embeddings)
    resultfilepath=os.path.join(resultpath,f"{filename}_index")
    if os.path.exists(resultfilepath):
        pass
    else:
        pass
    faissdb.save_local(resultfilepath)
    return filename

def merge_faiss(filepaths:[],resultpath=None,name=None):
    if name==None:
        name="faiss_index"    
    if resultpath==None:
        resultpath=os.path.dirname(filepaths[0])

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    
    faissdbs=[]
    faissdb_old=None
    for filepath in filepaths:
        try:
            faissdb = FAISS.load_local(filepath)
            faissdbs.append(filepath)
        except:
            continue
        if faissdb_old==None:
            faissdb_old=faissdb
        else:
            faissdb_old.merge_from(faissdb)
    result_faiss_path = os.path.join(resultpath,name)
    faissdb_old.save_local(result_faiss_path)
    
    return faissdbs