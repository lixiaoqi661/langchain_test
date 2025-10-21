from langchain_core.documents import Document
from load_model.a import MyVLLMLLM
from langchain_experimental.graph_transformers import LLMGraphTransformer
import asyncio
my_llm=MyVLLMLLM(api_url="http://117.50.27.204:8000")

llm_transformer_filtered = LLMGraphTransformer(
    llm=my_llm,
    allowed_nodes=["Recipe", "Foodproduct"],
    allowed_relationships=["CONTAINS"],
)


text = """
我最喜欢的烹饪创作是让人无法抗拒的 Vegan Chocolate Cake Recipe。这个美味的甜点以其浓郁的可可风味和柔软湿润的口感而闻名。它完全是素食、无乳制品的，并且由于使用了特殊的无麸质面粉混合物，也是无麸质的。
要制作这个蛋糕，食谱包含以下食品及其相应数量：250克无麸质面粉混合物、80克高品质可可粉、200克砂糖和10克发酵粉。为了丰富口感和确保完美发酵，食谱还包含5克香草精。在液体成分中，需要240毫升杏仁奶和60毫升植物油。
这个食谱可以制作一个巧克力蛋糕，被视为类型为甜点的 Foodproduct。
"""
documents = [Document(page_content=text)]
graph_documents_filtered =  asyncio.run(llm_transformer_filtered.aconvert_to_graph_documents(documents))
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")

