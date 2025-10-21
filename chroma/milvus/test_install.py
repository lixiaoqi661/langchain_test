from pymilvus import MilvusClient


def test_01():    # OK
    # 连接到 3221 端口
    client = MilvusClient(uri="http://localhost:3221")
    #
    # client.create_collection(
    #     collection_name="demo_collection",
    #     dimension=768,
    # )
    print(client.describe_collection("demo_collection"))
    conn=client.get("demo_collection",ids=[1,2,3])
    # print("集合创建成功！",conn.count(0))

    print(conn)
def test_02():
    pass

if __name__ == '__main__':
    pass
