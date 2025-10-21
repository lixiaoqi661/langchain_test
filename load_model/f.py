from langchain.text_splitter import MarkdownTextSplitter
from typing import List

def markdown_semantic_chunking(
    markdown_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    length_function=len # 默认使用字符长度，也可以是token长度
) -> List[str]:
    """
    使用 Markdown 文本分割器进行基于 Markdown 结构的分块。
    它会尝试保持 Markdown 标题、列表等的完整性。

    Args:
        markdown_text (str): 待分块的 Markdown 格式文本。
        chunk_size (int): 每个块的最大字符数。
        chunk_overlap (int): 相邻块之间的重叠字符数。
        length_function: 用于计算块长度的函数，默认为 len (字符数)。

    Returns:
        List[str]: 分割后的文本块列表。
    """
    # MarkdownTextSplitter 继承自 RecursiveCharacterTextSplitter
    # 它预设了一些适合 Markdown 的分隔符
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    chunks = text_splitter.split_text(markdown_text)
    return chunks

# 示例使用
if __name__ == "__main__":
    markdown_content = """
# 标题一：LangChain 简介

LangChain是一个用于开发由语言模型驱动的应用程序的框架。

## 子标题 1.1：核心功能

*   **组件:** 提供模块化抽象。
*   **现成链:** 简化常见用例。

# 标题二：文本分块的重要性

分块（Chunking）是将长文本分割成更小、更易于管理的部分的过程。
这对于将文本传递给语言模型或将其存储在向量数据库中至关重要。

## 子标题 2.1：分块策略

1.  固定大小分块
2.  基于语义结构分块
3.  滑动窗口分块
    """

    print("--- Markdown 语义分块 ---")
    markdown_chunks = markdown_semantic_chunking(markdown_content, chunk_size=150, chunk_overlap=30)
    for i, chunk in enumerate(markdown_chunks):
        print(f"Chunk {i+1} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 20)
    print("\n")
    