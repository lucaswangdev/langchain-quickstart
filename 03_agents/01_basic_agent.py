"""
第三课：Tools - 让 LLM 调用外部功能
==========================================
学习定义和使用 Tools

运行：python 01_basic_agent.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例 1：自定义 Tool
# ============================================
print("【示例 1】自定义 Tool")
print("-" * 30)

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "东京": "雨，22°C",
    }
    return weather_data.get(city, "未知城市")

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

# 列出可用工具
print("可用工具：")
for t in [get_weather, calculator]:
    print(f"  - {t.name}: {t.description}")

print(f"\n调用 weather 工具：北京")
result = get_weather.invoke("北京")
print(f"结果：{result}")

print(f"\n调用 calculator 工具：(10 + 5) * 2")
result = calculator.invoke("(10 + 5) * 2")
print(f"结果：{result}")

# ============================================
# 示例 2：带工具调用的对话
# ============================================
print("\n" + "=" * 30)
print("【示例 2】让 LLM 决定调用工具")
print("-" * 30)

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools([get_weather, calculator])

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，可以使用工具来回答问题。

可用工具：
- get_weather: 获取城市天气
- calculator: 执行数学计算

当用户询问天气或计算时，自动调用相应工具。"""),
    ("user", "{input}")
])

# 创建 chain
chain = prompt | llm_with_tools

print("\n问题：北京的天气怎么样？")
response = chain.invoke({"input": "北京的天气怎么样？"})
print(f"AI 消息类型：{type(response)}")
print(f"AI 消息内容：{response}")

# 检查是否有工具调用
if hasattr(response, 'tool_calls') and response.tool_calls:
    print(f"\n检测到工具调用：{response.tool_calls}")
    for tool_call in response.tool_calls:
        print(f"  工具：{tool_call['name']}")
        print(f"  参数：{tool_call['args']}")

# ============================================
# 示例 3：完整工具调用循环
# ============================================
print("\n" + "=" * 30)
print("【示例 3】完整工具调用循环")
print("-" * 30)

from langchain_core.runnables import RunnablePassthrough

def run_tool_and_get_response(tool_call, llm_with_tools):
    """执行工具并获取 LLM 响应"""
    # 执行工具
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    if tool_name == 'get_weather':
        result = get_weather.invoke(tool_args['city'])
    elif tool_name == 'calculator':
        result = calculator.invoke(tool_args['expression'])
    else:
        result = "未知工具"
    
    print(f"  -> 工具执行结果：{result}")
    return result

# 测试完整流程
print("\n问题：计算 (100 + 200) * 2")

# 1. 获取 LLM 响应
response = chain.invoke({"input": "计算 (100 + 200) * 2"})

if hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
        result = run_tool_and_get_response(tool_call, llm_with_tools)
        
        # 2. 执行工具后，让 LLM 生成最终响应
        # 需要将工具结果传回给 LLM
        messages = [
            HumanMessage(content="计算 (100 + 200) * 2"),
            response,
            ToolMessage(content=result, tool_call_id=tool_call['id'])
        ]
        
        final_response = llm.invoke(messages)
        print(f"\n最终响应：{final_response.content}")
else:
    print(f"响应：{response.content}")
