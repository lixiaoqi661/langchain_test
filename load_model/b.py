from langchain.llms.base import LLM
from typing import Optional, List, Any
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool,initialize_agent,AgentType
from a import MyVLLMLLM
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory,ConversationSummaryMemory



my_llm=MyVLLMLLM(api_url="http://10.128.7.115:8000")
my_llm._llm_type
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
buffer_memory  = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 2. ConversationBufferWindowMemory（只记住最近N轮对话）
window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True,
    k=2  # 只记住最近2轮对话
)

# 3. ConversationSummaryMemory（总结历史对话，节省token）
summary_memory = ConversationSummaryMemory(
    llm=my_llm,
    memory_key="chat_history",
    return_messages=True
)

# agent=initialize_agent(
#     tools=tools,
#     llm=my_llm,
#     agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True,
#     memory=memory1
# )

# # 测试智能体
# test_questions = [
#     "帮我计算一下 25 * 4 + 100",
#     "今天天气怎么样？",
#     "我想知道15除以3等于多少，然后搜索一下时间",  # 需要用两个工具
# ]

# print("=== 智能体测试开始 ===")
# for question in test_questions:
#     print(f"\n问题: {question}")
#     try:
#         answer = agent.run(question)
#         print(f"答案: {answer}")
#     except Exception as e:
#         print(f"错误: {e}")
#     print("-" * 50)

# print("================")
# print("记忆测试")
# response1 =agent.run("我叫张三  帮我计算10+5")
# print(response1 )

# print("第二次测试")
# response2=agent.run("还记得我叫什么名字嘛")
# print(response2)

# 测试不同Memory
memory_types = [
    ("完整记忆", buffer_memory),
    ("窗口记忆(只记2轮)", window_memory),
    ("摘要记忆", summary_memory)
]

for name, mem in memory_types:
    print(f"\n=== 测试 {name} ===")
    
    agent = initialize_agent(
        tools=[calc_tool, search_tool],
        llm=my_llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=mem,
        verbose=False,  # 关闭详细输出，方便观察
        handle_parsing_errors=True
    )
    
    # 进行多轮对话测试
    agent.run("我叫小明，我喜欢吃苹果")
    agent.run("帮我计算 5*6")  
    agent.run("搜索一下天气")
    
    # 测试记忆效果
    response = agent.run("你还记得我的名字和爱好吗？")
    print(f"{name}的记忆效果: {response}")
