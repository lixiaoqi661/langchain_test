import os
import json
import requests
from typing import List, Dict, Any
from pathlib import Path

from elasticsearch import Elasticsearch
import openai
import numpy as np
import datetime

class SimpleRAGSystem:
    def __init__(self, 
                 es_host: str = "http://localhost:9200",
                 ollama_host: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text",
                 openai_api_key: str = "",
                 index_name: str = "rag_documents"):
        """
        简化版RAG系统 - 使用远程Ollama嵌入模型
        
        Args:
            es_host: Elasticsearch服务器地址
            ollama_host: Ollama服务器地址
            embedding_model: Ollama嵌入模型名称
            openai_api_key: OpenAI API密钥
            index_name: Elasticsearch索引名称
        """
        # 初始化Elasticsearch
        self.es = es = Elasticsearch(
                        "http://117.50.27.204:1200", # 你的 Elasticsearch 地址
                        basic_auth=("elastic", "infini_rag_flow"), # 替换为你的用户名和密码或 API key
                        # 或者用 Basic Auth
                        # basic_auth=("elastic", "your_password"),
                         verify_certs=False, # 如果是本地测试环境且没有正确配置 SSL 证书，可以设置为 False
                        ssl_show_warn=False
                    )
        self.index_name = index_name
        
        # 初始化Ollama配置
        self.ollama_host = ollama_host.rstrip('/')
        self.embedding_model = embedding_model
        
        # 测试Ollama连接并获取模型信息
        self.embedding_dim = self._init_ollama_embedding()
        
        # 设置OpenAI
        # openai.api_key = openai_api_key
        
        # 创建索引
        self._create_index()
        
        print("RAG系统初始化完成!")
    
    def _init_ollama_embedding(self):
        """初始化Ollama嵌入模型并获取向量维度"""
        try:
            # 测试连接
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                raise Exception(f"无法连接到Ollama服务器: {self.ollama_host}")
            
            # 检查模型是否存在
            models = response.json().get('models', [])
            model_names = [model['name'].split(':')[0] for model in models]
            
            if self.embedding_model not in model_names:
                print(f"模型 {self.embedding_model} 不存在，正在拉取...")
                self._pull_model(self.embedding_model)
            
            # 获取向量维度
            test_embedding = self._get_embedding("test")
            embedding_dim = len(test_embedding)
            
            print(f"Ollama连接成功! 使用模型: {self.embedding_model}, 向量维度: {embedding_dim}")
            return embedding_dim
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama连接失败: {e}")
        except Exception as e:
            raise Exception(f"初始化Ollama嵌入模型失败: {e}")
    
    def _pull_model(self, model_name: str):
        """拉取Ollama模型"""
        try:
            print(f"正在拉取模型 {model_name}，请稍候...")
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'status' in data:
                        print(f"拉取状态: {data['status']}")
                    if data.get('status') == 'success':
                        print(f"模型 {model_name} 拉取成功!")
                        break
        except Exception as e:
            raise Exception(f"拉取模型失败: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """使用Ollama获取文本嵌入向量"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            print(f'------{response}')
            if response.status_code != 200:
                raise Exception(f"Ollama API调用失败: {response.status_code}")
            
            result = response.json()
            return result['embedding']
            
        except requests.exceptions.Timeout:
            raise Exception("Ollama API调用超时")
        except Exception as e:
            raise Exception(f"获取嵌入向量失败: {e}")
    
    def _create_index(self):
        """创建Elasticsearch索引"""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim
                    },
                    "source": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        # 如果索引不存在则创建
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"创建索引: {self.index_name}")
    
    def add_text(self, text: str, source: str = "manual_input") -> str:
        """添加文本到知识库"""
        try:
            # 文本分块
            chunks = self._split_text(text)
            
            added_count = 0
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # 跳过太短的块
                    continue
                
                print(f"正在处理第 {i+1} 个文本块...")
                
                # 使用Ollama生成嵌入向量
                embedding = self._get_embedding(chunk)
                
                # 存储到ES
                doc = {
                    "content": chunk,
                    "embedding": embedding,
                    "source": f"{source}_chunk_{i}",
                    "timestamp":  datetime.datetime.now()
                }
                
                self.es.index(index=self.index_name, body=doc)
                added_count += 1
            
            self.es.indices.refresh(index=self.index_name)
            return f"成功添加 {added_count} 个文本块到知识库"
            
        except Exception as e:
            return f"添加文本时出错: {str(e)}"
    
    def add_document(self, file_path: str) -> str:
        """添加本地文档到知识库"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return f"文件不存在: {file_path}"
            
            # 读取文件内容
            content = self._read_file(file_path)
            if not content:
                return f"无法读取文件内容: {file_path}"
            
            # 添加到知识库
            result = self.add_text(content, source=file_path.name)
            return f"文档 {file_path.name}: {result}"
            
        except Exception as e:
            return f"添加文档时出错: {str(e)}"
    
    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix in ['.txt', '.md', '.py', '.js', '.html', '.css']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                print(f"不支持的文件格式: {suffix}")
                return ""
                
        except Exception as e:
            print(f"读取文件出错: {e}")
            return ""
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """简单的文本分块"""
        # 按句子分割
        sentences = text.replace('\n\n', '\n').split('。')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果当前块加上新句子不超过限制，就添加
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + "。"
            else:
                # 否则保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关文档"""
        try:
            print(f"正在搜索: {query}")
            
            # 使用Ollama生成查询向量
            query_embedding = self._get_embedding(query)
            
            # ES向量搜索
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": top_k
            }
            
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'content': hit['_source']['content'],
                    'source': hit['_source']['source'],
                    'score': hit['_score']
                })
            
            return results
            
        except Exception as e:
            print(f"搜索时出错: {e}")
            return []
    
    def chat(self, question: str) -> str:
        """与智能体对话"""
        try:
            # 搜索相关内容
            search_results = self.search(question, top_k=3)
            
            if not search_results:
                return "抱歉，我在知识库中没有找到相关信息。"
            
            # 构建上下文
            context = "\n\n".join([
                f"文档片段 {i+1} (来源: {result['source']}, 相似度: {result['score']:.3f}):\n{result['content']}"
                for i, result in enumerate(search_results)
            ])
            
            # 构建提示词
            prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明。

上下文信息:
{context}

问题: {question}

请基于上下文信息给出准确、有帮助的回答:"""
            
            # 调用OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个有用的AI助手，会基于提供的上下文信息来回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500, 
                
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # 添加来源信息
            sources = list(set([result['source'] for result in search_results]))
            source_info = f"\n\n📚 参考来源: {', '.join(sources[:3])}"
            similarity_info = f"\n🔍 最高相似度: {search_results[0]['score']:.3f}"
            
            return answer + source_info + similarity_info
            
        except Exception as e:
            return f"对话时出错: {str(e)}"
    
    def chat_with_ollama(self, question: str, chat_model: str = "bge-m3") -> str:
        """使用Ollama进行对话 (可选功能)"""
        try:
            # 搜索相关内容
            search_results = self.search(question, top_k=3)
            
            if not search_results:
                return "抱歉，我在知识库中没有找到相关信息。"
            
            # 构建上下文
            context = "\n\n".join([
                f"文档片段 {i+1}:\n{result['content']}"
                for i, result in enumerate(search_results)
            ])
            
            # 构建提示词
            prompt = f"""基于以下上下文信息回答问题：
                    上下文信息:
                    {context}
                    问题: {question}
                    请基于上下文信息给出准确、有帮助的回答:
                    """
            
            # 调用Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": chat_model,
                    "prompt": prompt
                },
                timeout=60
            )
            print(response)
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '无法获取回答')
            else:
                answer = f"Ollama API调用失败: {response.status_code}"
            
            # 添加来源信息
            sources = list(set([result['source'] for result in search_results]))
            source_info = f"\n\n📚 参考来源: {', '.join(sources[:3])}"
            
            return answer + source_info
            
        except Exception as e:
            return f"使用Ollama对话时出错: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            # 获取文档总数
            count_response = self.es.count(index=self.index_name)
            total_docs = count_response['count']
            
            # 获取索引信息
            index_info = self.es.indices.stats(index=self.index_name)
            index_size = index_info['indices'][self.index_name]['total']['store']['size_in_bytes']
            
            return {
                "total_documents": total_docs,
                "index_size_mb": round(index_size / (1024 * 1024), 2),
                "index_name": self.index_name,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dim,
                "ollama_host": self.ollama_host
            }
            
        except Exception as e:
            return {"error": f"获取统计信息时出错: {str(e)}"}
    
    def clear_index(self):
        """清空知识库"""
        try:
            self.es.indices.delete(index=self.index_name)
            self._create_index()
            return "知识库已清空"
        except Exception as e:
            return f"清空知识库时出错: {str(e)}"
    
    def test_ollama_connection(self) -> str:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_list = [model['name'] for model in models]
                return f"Ollama连接正常! 可用模型: {model_list}"
            else:
                return f"Ollama连接失败: HTTP {response.status_code}"
        except Exception as e:
            return f"Ollama连接测试失败: {str(e)}"


# 使用示例和主程序
def main():
    print("=== RAG系统 - 使用远程Ollama嵌入模型 ===")
    print("请确保以下服务已启动:")
    print("1. Elasticsearch (默认: http://localhost:9200)")
    print("2. Ollama (默认: http://localhost:11434)")
    
    # 配置
    ES_HOST = "http://117.50.27.204:1200"
    OLLAMA_HOST = "http://117.50.27.204:11435"  # 可以是远程地址，如 "http://192.168.1.100:11434"
    EMBEDDING_MODEL = "bge-m3"    # 或其他嵌入模型如 "mxbai-embed-large"
    # OPENAI_API_KEY = ""  # 替换为你的API密钥
    
    try:
        # 初始化RAG系统
        rag = SimpleRAGSystem(
            es_host=ES_HOST,
            ollama_host=OLLAMA_HOST,
            embedding_model=EMBEDDING_MODEL,
            # openai_api_key=OPENAI_API_KEY
        )
        
        print("\n命令说明:")
        print("- add: 添加文本到知识库")
        print("- file: 添加本地文件到知识库")
        print("- test: 测试Ollama连接")
        print("- stats: 查看知识库统计")
        print("- clear: 清空知识库")
        print("- ollama: 使用Ollama模型对话")
        print("- quit: 退出程序")
        print("- 或者直接输入问题与AI对话")
        print("-" * 50)
        
        while True:
            user_input = input("\n请输入: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'add':
                text = input("请输入要添加的文本: ")
                result = rag.add_text(text)
                print(f"结果: {result}")
            elif user_input.lower() == 'file':
                file_path = input("请输入文件路径: ")
                result = rag.add_document(file_path)
                print(f"结果: {result}")
            elif user_input.lower() == 'test':
                result = rag.test_ollama_connection()
                print(f"连接测试: {result}")
            elif user_input.lower() == 'stats':
                stats = rag.get_stats()
                print(f"知识库统计: {stats}")
            elif user_input.lower() == 'clear':
                result = rag.clear_index()
                print(f"结果: {result}")
            elif user_input.lower() == 'ollama':
                question = input("请输入问题 (使用Ollama回答): ")
                print("Ollama正在思考...")
                answer = rag.chat_with_ollama(chat_model=EMBEDDING_MODEL,question=question)
                print(f"\nOllama回答:\n{answer}")
            elif user_input:
                # 与AI对话
                print("AI正在思考...")
                answer = rag.chat(user_input)
                print(f"\nAI回答:\n{answer}")
    
    except Exception as e:
        print(f"系统初始化失败: {e}")
        print("请检查:")
        print("1. Elasticsearch是否已启动")
        print("2. Ollama服务是否已启动")
        print("3. 嵌入模型是否已下载")
        print("4. 网络连接是否正常")

if __name__ == "__main__":
    main()


# # 配置示例文件
# def create_config_example():
#     """创建配置示例文件"""
#     config = {
#         "es_host": "http://localhost:9200",
#         "ollama_host": "http://localhost:11434",  # 可改为远程地址
#         "embedding_model": "nomic-embed-text",
#         "chat_model": "llama3.1",
#         "openai_api_key": "your-openai-api-key-here",
#         "index_name": "rag_documents"
#     }
    
#     with open("rag_config.json", "w", encoding="utf-8") as f:
#         json.dump(config, f, indent=2, ensure_ascii=False)
    
#     print("配置示例文件已创建: rag_config.json")

# 取消注释以创建配置文件
# create_config_example()