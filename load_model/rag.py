from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import OpenAIEmbeddings
from elasticsearch import Elasticsearch

# 纯ES连接测试
es = Elasticsearch( 
    "http://117.50.27.204:1200", # 你的 Elasticsearch 地址
    basic_auth=("elastic", "infini_rag_flow"), # 替换为你的用户名和密码或 API key
    # 或者用 Basic Auth
    # basic_auth=("elastic", "your_password"),
    verify_certs=False, # 如果是本地测试环境且没有正确配置 SSL 证书，可以设置为 False
    ssl_show_warn=False
)
print("ES连接状态:", es.ping())  # True表示连接成功
print("集群信息:", es.info())