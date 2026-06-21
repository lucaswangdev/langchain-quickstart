"""
第一课：Prompt 模板
====================
学习使用 PromptTemplate 管理复杂的 Prompt

运行：python 02_prompt_template.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import os

# 从项目根目录加载 .env
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例 1：简单模板
# ============================================
print("【示例 1】简单模板")
print("-" * 30)

template = PromptTemplate.from_template(
    "请把以下句子翻译成 {language}：{sentence}"
)

prompt = template.invoke({
    "language": "英文",
    "sentence": "你好，世界！"
})

print(f"生成的 Prompt：{prompt.to_string()}")

response = llm.invoke(prompt)
print(f"翻译结果：{response.content}")

# ============================================
# 示例 2：带系统提示词的模板
# ============================================
print("\n【示例 2】带系统提示词的模板")
print("-" * 30)

template_with_system = PromptTemplate.from_template(
    """你是一个专业的 {role}。
用户问题：{question}
请用专业的方式回答。"""
)

prompt = template_with_system.invoke({
    "role": "Python 编程导师",
    "question": "什么是装饰器？"
})

response = llm.invoke(prompt)
print(f"回答：{response.content}")

# ============================================
# 示例 3：列表格式的模板
# ============================================
print("\n【示例 3】格式化输出")
print("-" * 30)

template_list = PromptTemplate.from_template(
    """请列出 {topic} 的 {count} 个特点：

格式：
1. 特点1
2. 特点2
...
"""
)

prompt = template_list.invoke({
    "topic": "Python",
    "count": 3
})

response = llm.invoke(prompt)
print(response.content)
