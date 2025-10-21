from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool,initialize_agent
from langchain.callbacks.manager import CallbackManagerForLLMRun

class MyVLLMLLM(LLM):
    api_url: str # 你的vLLM服务地址
    model_name: str = "deepseek-32b-awq"  # 添加model参数

    @property
    def _llm_type(self) -> str:
        return "my_vllm"
    def clean_response(self, text: str) -> str:
        """清理回答中的思考内容，只保留最终答案"""
        import re
        
        # 移除完整的思考块（包括多行内容）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 清理多余的空行
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,**kwargs) -> str:
        # 这里需要你把之前的requests调用逻辑放进来
        # 把prompt发送给你的vLLM服务，返回生成的文本
        data = {
        "model": self.model_name, 
        "max_tokens": 5000,
        "tempertaure": 0.3,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "system", "content": "你是一个中文AI助手，请始终用中文回答，不要使用英文。"}
                    ]# 把prompt放这里
        }
        response = requests.post(f"{self.api_url}/v1/chat/completions",
                                  json=data,
                                  headers={"Content-Type": "application/json"})
    
        if response.status_code != 200:
            print(f"请求失败: {response.text}")
            return "请求失败"  # 返回一个默认字符串，而不是None
        
        try:
            result = response.json()
            
            # 检查返回的数据结构
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                answer = self.clean_response(answer)
                return answer if answer else "空回答"  # 确保不返回None
            else:
                return "未找到回答"
        except Exception as e:
            print(f"解析JSON失败: {e}")
            return "解析失败"
# my_llm=MyVLLMLLM(api_url="http://117.50.27.204:8000")
# outline_template = """你是一个小说写作专家，请写一个{story_type}小说的故事大纲，包含主要人物情节:

# 请包含以下要素：
# 1. **主要人物**（2-3个核心角色，包含姓名和性格特点）
# 2. **故事背景**（时间、地点、世界设定）
# 3. **主要冲突**（故事的核心矛盾）
# 4. **情节结构**（开头-发展-高潮-结局的简要描述）

# 字数：300-500字
# 风格：{story_type}类型的经典特征

# 小说大纲："""

# outline_prompt = PomptTemplate(
#     input_variables=["story_type"],
#     template=outline_template
# )

# # 第二步：给出解决方案
# chapter_template = """根据以下故事大纲，写出第一章内容：\n{outline}:"""
# chapter_prompt = PromptTemplate(
#     input_variables=["outline"],
#     template=chapter_template
# )

# # 创建两个链
# outline_chain = LLMChain(llm=my_llm, prompt=outline_prompt)
# chapter_chain = LLMChain(llm=my_llm, prompt=chapter_prompt)


# story_type="玄幻"
# print("=== 开始链式调用 ===")
# print(f"原始问题: {story_type}")
# print("\n--- 第一步：写小说大纲 ---")

# # 第一步：分析问题
# outline_result = outline_chain.run(story_type=story_type)
# print(f"分析结果:\n{outline_result}")

# print("\n--- 第二步：写第一章 ---")

# # 第二步：基于分析给出解决方案
# chapter_result = chapter_chain.run(outline=outline_result)
# print(f".解决方案:\n{chapter_result}")

# print("\n=== 链式调用完成 ===")