from dataclasses import fields
from pydoc import describe

import numpy as np
from pymilvus import MilvusClient, DataType

# 连接到 Docker 中的 Milvus 服务
client = MilvusClient(uri="http://localhost:3221")

def test_create():
    # 创建数据库（可选）
    client.create_collection(
        collection_name="demo_2",
        dimension=128,
        auto_id=False,
        index_type="IVF_FLAT",
        metric_type="IP",
        fields=[
            {"name": "id", "type": "INT64", "is_primary": True, "auto_id": False},
            {"name": "vector", "type": "FLOAT_VECTOR", "dimension": 128},
            {"name": "text", "type": "VARCHAR", "max_length": 512},
            {"name": "metadata", "type": "JSON"}
        ],
    )


def test_create_1():
    # 定义 schema
    schema = client.create_schema(
        auto_id=False,  # 自动生成 ID
        enable_dynamic_field=True  # 支持动态字段
    )
    # 添加字段
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    collection_name = "demo_collection",
    index_params={
        "field_name":"vector",
        "index_type":"IVF_FLAT",  # 索引类型
        "metric_type":"L2",  # 距离度量: L2, IP, COSINE
        "params":{"nlist": 128}
    }
    # 创建索引参数
    # client.create_collection_with_schema(
    #     index_params=index_params,
    #     collection_name=collection_name
    # )
    # 创建集合
    client.drop_collection("demo_collection01")
    client.create_collection_with_schema(
        collection_name="demo_collection01",
        schema=schema,
        index_param=index_params
    )
    # print(f"Collection '{collection_name}' created successfully")

def test_insert():
    collection_name = "demo_collection01"
    # 模拟生成 3 条数据
    vectors = np.random.rand(3, 128).tolist()
    texts = [
        "一只小猫在晒太阳",
        "一只狗在草地上跑",
        "夕阳下的城市风景"
    ]

    metadata_list = [
        {"source": "weibo", "tag": "cat", "timestamp": 1730000000},
        {"source": "weibo", "tag": "dog", "timestamp": 1730000001},
        {"source": "weibo", "tag": "city", "timestamp": 1730000002},
    ]

    # 组装成插入格式
    data = [
        {
            # "id": i + 1,
            "vector": vectors[i],
            "text": texts[i],
            "metadata": metadata_list[i]
        }
        for i in range(3)
    ]

    # 插入
    res = client.insert(collection_name=collection_name, data=data)
    print("Inserted:", res)
    pass



def test_query():
    client.flush(collection_name="demo_collection01")
    res=client.query(
        collection_name="demo_collection01",
        filter="'小猫' in text",  # 限制比较多， 就不用这个做文本检索，文本检索用es，，这个主要是向量检索
        limit=5
    )
    print(res)
    pass

def test_search():
    client.search(
        collection_name="demo_collection01",

    )
    pass
