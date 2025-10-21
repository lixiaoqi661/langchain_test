import os
from dotenv import load_dotenv
import mysql.connector
from py2neo import Graph, Node, Relationship

# 读取 .env 文件
load_dotenv(".env")

# MySQL 配置
mysql_config = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

# Neo4j 配置
neo4j_config = {
    "uri": os.getenv("NEO4J_URI"),
    "user": os.getenv("NEO4J_USER"),
    "password": os.getenv("NEO4J_PASSWORD")
}

# 建立连接
mysql_conn = mysql.connector.connect(**mysql_config)
mysql_cursor = mysql_conn.cursor(dictionary=True)
graph = Graph(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))



