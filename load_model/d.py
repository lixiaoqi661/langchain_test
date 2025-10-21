from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def fixed_size_chunking_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    使用递归字符文本分割器进行固定大小（带重叠）分块。
    它会尝试按段落、句子等分隔符分割，如果块仍太大，则回退到字符分割。

    Args:
        text (str): 待分块的原始文本。
        chunk_size (int): 每个块的最大字符数。
        chunk_overlap (int): 相邻块之间的重叠字符数。
        separators (List[str], optional): 用于分割的字符列表，优先级从高到低。
                                            默认为 None，使用 ['\n\n', '\n', ' ', '']。

    Returns:
        List[str]: 分割后的文本块列表。
    """
    if separators is None:
        # 默认分隔符，优先按段落、换行、空格分割
        # 如果块仍然太大，最终会按字符分割（''）
        separators = ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,  # 使用 len() 计算长度，对于字符数
        is_separator_regex=False # 如果分隔符是正则表达式则设为 True
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 示例使用
if __name__ == "__main__":
    long_text = """
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

    print("--- 固定大小（递归字符）分块 ---")
    fixed_chunks = fixed_size_chunking_recursive(long_text, chunk_size=150, chunk_overlap=30)
    for i, chunk in enumerate(fixed_chunks):
        print(f"Chunk {i+1} (长度: {len(chunk)}):")
        print(chunk)
        print("-" * 20)
    print("\n")