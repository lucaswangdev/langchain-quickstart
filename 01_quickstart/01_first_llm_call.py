"""
第一课：第一个 LLM 调用
===========================
这是你使用 LangChain 的第一个例子！

运行：python 01_first_llm_call.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# 从项目根目录加载 .env
from pathlib import Path
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

# 创建 LLM 实例
model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# 调用 LLM
response = llm.invoke("Hello, World!")

# 打印回复内容
print("=" * 50)
print("LLM 回复：")
print(response.content)
print("=" * 50)
