from operator import contains

import chromadb
from intake.readers import metadata_fields

'''
上面的代码中，我们向Chroma提交了两个文档（简单起见，是两个字符串），一个是This is a document about engineer，一个是This is a document about steak。
若在add方法没有传入embedding参数，则会使用Chroma默认的all-MiniLM-L6-v2 方式进行embedding。随后，我们对数据集进行query，要求返回两个最相关的结果。
提问内容为：Which food is the best?
'''
def test_01():
    client = chromadb.PersistentClient(path="E:/code/langchain_test/db")
    collection = client.get_or_create_collection(name="collection")
    collection.add(
        documents=["This is a document about engineer", "This is a document about steak"],
        metadatas=[{"source": "doc1"}, {"source": "doc2"}],
        ids=["id1","id2"]
    )
    total_count = collection.count()
    print(f"Collection '{collection.name}' 中共有 {total_count} 条数据。")
    results=collection.query(
        query_texts=["which food is best"],
        where={"metadata_field":{"$eq":"engineer"}},
        n_results=2,
        ids=["id1"]
    )
    # print(results)
    if total_count > 0:
        all_data = collection.get(
            limit = total_count,
            where_document={"$contains":"about"},
            include = ["metadatas", "documents"], # 指定需要返回的内容
        )
        print("\n--- 开始显示所有数据 ---")

        # 4. 格式化并打印结果，这样更易读
        # all_data 是一个字典，我们遍历其中的 ids 列表来逐条打印
        ids = all_data['ids']
        documents = all_data['documents']
        metadatas = all_data['metadatas']
        for i in range(len(ids)):
            print(f"--- 条目 {i + 1} ---")
        print(all_data)
        print("-" * 20)
    else:
        print("Collection 中没有数据。")
def test_02():

    # collection.add(
    #     documents=[],
    #     metadatas=[],
    #     ids=[]
    # )
    # collection.query(
    #
    # )
    pass
