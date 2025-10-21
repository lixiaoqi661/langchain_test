from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict, Any
import uuid

def parent_child_chunking(
    text: str,
    parent_chunk_size: int = 1000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 200,
    child_chunk_overlap: int = 50,
    separators: List[str] = None
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    实现父-子分块策略。生成用于检索的子块和提供上下文的父块，并建立映射。

    Args:
        text (str): 待分块的原始文本。
        parent_chunk_size (int): 父块的最大字符数。
        parent_chunk_overlap (int): 父块的重叠字符数。
        child_chunk_size (int): 子块的最大字符数。
        child_chunk_overlap (int): 子块的重叠字符数。
        separators (List[str], optional): 用于分割的字符列表。

    Returns:
        Tuple[List[str], List[str], Dict[str, str]]:
            - List[str]: 所有父块。
            - List[str]: 所有子块。
            - Dict[str, str]: 子块 ID 到父块 ID 的映射。
                              （实际应用中，子块通常会存储其父块的引用或 ID）
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    # 1. 生成父块 (Parent Chunks)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=separators,
        length_function=len
    )
    parent_chunks = parent_splitter.split_text(text)

    # 2. 为每个父块生成子块，并建立映射
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=separators,
        length_function=len
    )

    all_child_chunks = []
    # 这里存储子块到父块内容的映射，实际应用中会是子块到父块ID
    child_to_parent_map = {}
    
    # 辅助字典，存储父块内容到其 ID 的映射，方便查找
    parent_content_to_id: Dict[str, str] = {}
    
    # 存储父块 ID 列表 (如果父块需要有唯一 ID)
    parent_ids: List[str] = []

    for parent_chunk in parent_chunks:
        parent_id = str(uuid.uuid4()) # 为每个父块生成唯一ID
        parent_ids.append(parent_id)
        parent_content_to_id[parent_chunk] = parent_id
        
        # 将父块作为原始文本，再分割成子块
        current_child_chunks = child_splitter.split_text(parent_chunk)
        
        for child_chunk in current_child_chunks:
            all_child_chunks.append(child_chunk)
            # 记录子块内容（或其hash/ID）与父块内容的映射
            # 实际应用中，子块通常会直接存储父块的ID
            child_to_parent_map[child_chunk] = parent_chunk # 这里直接映射到父块内容
            # 如果是存储 ID，则 child_to_parent_map[child_chunk_id] = parent_id

    # 在实际应用中，你会将 all_child_chunks 嵌入并存储到向量数据库。
    # 当检索到某个 child_chunk 后，你会通过 child_to_parent_map 找到对应的 parent_chunk，
    # 并将其作为上下文传递给 LLM。
    
    # 为了演示，这里返回父块列表，子块列表，以及子块内容到父块内容的映射
    return parent_chunks, all_child_chunks, child_to_parent_map

# 示例使用
if __name__ == "__main__":
    long_text_for_parent_child = """
    LangChain是一个用于开发由语言模型驱动的应用程序的框架。
    它的核心是一个可以链接不同组件以创建更高级别用例的工具。
    LangChain的主要价值主张包括：
    1. 组件：为使用大型语言模型构建应用程序提供了模块化抽象集。
    2. 现成链：提供结构化的“链”，用于常见的用例，如文档问答、聊天机器人等。

    分块（Chunking）是将长文本分割成更小、更易于管理的部分的过程。
    这对于将文本传递给语言模型或将其存储在向量数据库中至关重要。
    适当的分块可以显著影响检索系统的性能。

    常见的策略包括固定大小分块、基于语义结构分块和滑动窗口分块。
    选择正确的分块策略取决于具体的应用场景和数据特性。
    """
    
    print("--- 父-子分块 ---")
    parent_chunks, child_chunks, child_to_parent_map = parent_child_chunking(
        long_text_for_parent_child,
        parent_chunk_size=300,  # 父块较大
        parent_chunk_overlap=50,
        child_chunk_size=80,   # 子块较小用于检索
        child_chunk_overlap=20
    )

    print(f"生成的父块数量: {len(parent_chunks)}")
    print(f"生成的子块数量: {len(child_chunks)}")
    print("\n前两个父块:")
    for i, chunk in enumerate(parent_chunks[:2]):
        print(f"Parent Chunk {i+1} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 20)

    print("\n前三个子块及其对应的父块（内容而非ID用于演示）:")
    for i, child_chunk in enumerate(child_chunks[:3]):
        parent_content = child_to_parent_map.get(child_chunk, "N/A")
        print(f"Child Chunk {i+1} (长度: {len(child_chunk)}):")
        print(child_chunk)
        print(f"对应的 Parent Content (长度: {len(parent_content)}):")
        print(parent_content)
        print("-" * 20)
    print("\n")