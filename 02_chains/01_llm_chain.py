"""
第二课：LCEL Chain - 使用新版 LangChain
=========================================
LangChain v1.x 使用 LCEL (LangChain Expression Language)
通过 | 管道操作符组合组件

运行：python 01_llm_chain.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例 1：简单的概念解释器
# ============================================
print("【示例 1】概念解释器")
print("-" * 30)

template = PromptTemplate.from_template(
    "请用一句话解释以下概念：{concept}"
)

# 使用 LCEL | 管道操作符组合
chain = template | llm

result = chain.invoke({"concept": "机器学习"})
print(f"概念：机器学习")
print(f"解释：{result.content}")

# ============================================
# 示例 2：多输入的 Chain
# ============================================
print("\n【示例 2】多输入 Chain")
print("-" * 30)

template_multi = PromptTemplate.from_template(
    """作为一位{role}，请评价以下代码：

```{language}
{code}
```

简要评价："""
)

chain_multi = template_multi | llm

result = chain_multi.invoke({
    "role": "Python 专家",
    "language": "python",
    "code": "print('Hello')"
})

print(f"代码评价：{result.content}")

# ============================================
# 示例 3：翻译 Chain
# ============================================
print("\n【示例 3】翻译 Chain")
print("-" * 30)

template_translate = PromptTemplate.from_template(
    "将以下中文翻译成英文：{chinese_text}"
)

chain_translate = template_translate | llm

test_text = "LangChain 让 LLM 开发变得简单"
result = chain_translate.invoke({"chinese_text": test_text})

print(f"原文：{test_text}")
print(f"译文：{result.content}")

# ============================================
# 示例 4：带输出解析的 Chain
# ============================================
print("\n【示例 4】带结构化输出的 Chain")
print("-" * 30)

from langchain_core.output_parsers import StrOutputParser

template_json = PromptTemplate.from_template(
    """请用 JSON 格式回答：
    问题：{question}
    
    JSON格式：{{"answer": "你的答案"}}"""
)

chain_with_parser = template_json | llm | StrOutputParser()

result = chain_with_parser.invoke({"question": "什么是 Python？"})
print(f"结构化输出：{result}")
