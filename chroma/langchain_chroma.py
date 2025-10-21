from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings

from load_model.a import MyVLLMLLM

#读取原始文档
loader=TextLoader(file_path='files/西游记.txt',encoding='utf-8').load()

#分割文档
text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=100)
docs=text_spliter.split_documents(loader)

#加载ollama 的embedding模型
my_embd=OllamaEmbeddings(
    model="bge-m3",
    base_url="http://10.128.7.115:3103"
)


#向量化写入chromadb
db=Chroma.from_documents(documents=docs,
                         embedding=my_embd,
                         persist_directory="db",
                         collection_name="my_collection"
                         )

#持久化
db.persist()


# my_llm=MyVLLMLLM(api_url="http://10.128.7.115:3101")


# # 重新加载
# db1=Chroma(persist_directory="db",embedding_function=embd)
#
# #构建问答系统
# retriver = db1.as_retriever(search_kwargs={"k":3})
# qa=RetrievalQA.from_chain_type(
#     llm=sd
#



