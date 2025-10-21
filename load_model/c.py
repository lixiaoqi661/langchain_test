from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool,initialize_agent,AgentType
from a import MyVLLMLLM
from langchain.memory import ConversationBufferMemory


llm=MyVLLMLLM(api_url="http://117.50.27.204:8000")

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 你的任务：实现这个函数
        # 输入：像"2+3*4"这样的字符串
        # 输出：计算结果的字符串
        result=eval(expression)
        return   result
    except Exception as e:
        return f"计算错误: {e}"

# 创建工具
calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description="用于计算数学表达式，输入如：2+3*4"
)

def fake_search(query: str) -> str:
    """模拟搜索功能"""
    query = query.lower()  # 转小写方便匹配
    
    if "天气" in query:
        return "今天晴天，温度25度，适合外出活动"
    elif "时间" in query:
        return "现在是2024年下午3点30分"
    elif "新闻" in query:
        return "最新新闻：科技公司发布了新的AI产品"  
    elif "股票" in query or "股价" in query:
        return "当前股市表现平稳，主要指数小幅上涨"
    elif "餐厅" in query or "美食" in query:
        return "推荐附近的川菜馆：麻辣香锅，评分4.5星"
    else:
        return f"搜索到了关于'{query}'的相关信息：这是一个模拟的搜索结果"
    

# 包装成工具
search_tool = Tool(
    name="Search", 
    func=fake_search,
    description="用于搜索信息，比如天气、时间、新闻等"
)

tools=[
    calc_tool,
    search_tool
]

# 创建Memory对象
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# agent=initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     memory=memory
# )
 
 