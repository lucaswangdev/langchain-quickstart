"""
第四课：对话 Memory - 记住对话历史
====================================
让 LLM 记住对话上下文

运行：python 01_conversation_memory.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例 1：基础对话 Memory（手动管理）
# ============================================
print("【示例 1】手动 Memory 管理")
print("-" * 30)

# 对话历史
chat_history = []

# 第一轮对话
print("用户：我叫小明，程序员")
chat_history.append(HumanMessage(content="我叫小明，程序员"))

prompt1 = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])
chain1 = prompt1 | llm | StrOutputParser()
response1 = chain1.invoke({"input": "我叫小明，程序员"})
chat_history.append(AIMessage(content=response1))
print(f"AI：{response1}")

# 第二轮对话
print("\n用户：你叫什么？")
chat_history.append(HumanMessage(content="你叫什么？"))
prompt2 = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])
chain2 = prompt2 | llm | StrOutputParser()
response2 = chain2.invoke({"input": "你叫什么？"})
chat_history.append(AIMessage(content=response2))
print(f"AI：{response2}")

# 第三轮对话（会记得我是程序员）
print("\n用户：我的职业是什么？")
chat_history.append(HumanMessage(content="我的职业是什么？"))

# 使用完整的对话历史
prompt3 = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的 AI 助手。请基于对话历史回答问题。"),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])

chain3 = prompt3 | llm | StrOutputParser()
response3 = chain3.invoke({
    "chat_history": chat_history[:-1],  # 不包括最后一轮用户输入
    "input": "我的职业是什么？"
})
chat_history.append(AIMessage(content=response3))
print(f"AI：{response3}")

print("\n" + "-" * 30)
print("对话历史（最后 4 条消息）：")
for msg in chat_history[-4:]:
    role = "用户" if isinstance(msg, HumanMessage) else "AI"
    print(f"  {role}：{msg.content[:30]}...")

# ============================================
# 示例 2：使用 LCEL 管理历史
# ============================================
print("\n" + "=" * 30)
print("【示例 2】使用 LCEL withMessagesPlaceholder")
print("-" * 30)

chat_history2 = []

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的 AI 助手。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}")
])

chain_with_history = (
    RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history", []))
    | prompt_template
    | llm
    | StrOutputParser()
)

def add_and_get(chain, history, user_input):
    """添加用户消息，运行 chain，返回响应"""
    history.append(HumanMessage(content=user_input))
    response = chain.invoke({"input": user_input, "chat_history": history[:-1]})
    history.append(AIMessage(content=response))
    return response

print("用户：我是程序员")
resp = add_and_get(chain_with_history, chat_history2, "我是程序员")
print(f"AI：{resp}")

print("\n用户：我叫什么？")
resp = add_and_get(chain_with_history, chat_history2, "我叫什么？")
print(f"AI：{resp}")
