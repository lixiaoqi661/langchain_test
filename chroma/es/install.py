# es安装
def test_connect():
    from elasticsearch import Elasticsearch

    # ES 容器地址
    ES_HOST = "http://localhost:9200"

    # 如果开启了安全认证，需要加用户名密码
    # client = Elasticsearch(ES_HOST, basic_auth=("elastic", "your_password"))

    client = Elasticsearch(ES_HOST)

    try:
        # 测试集群是否可用
        if client.ping():
            print("✅ Elasticsearch 连接成功！")
        else:
            print("❌ Elasticsearch 连接失败")
    except Exception as e:
        print("❌ 连接异常:", e)

    # 查询版本信息
    try:
        info = client.info()
        print("Elasticsearch 信息:", info)
    except Exception as e:
        print("获取 info 异常:", e)


def test_connect_1():
    from elasticsearch import Elasticsearch

    client = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", "changeme123"),
        verify_certs=False  # 本地测试可关闭证书验证
    )

    # 测试连接
    if client.ping():
        print("✅ 连接成功")
    else:
        print("❌ 连接失败")

    # 查看版本信息
    print(client.info())


def test_insert():

    pass
def test_select():

    pass
