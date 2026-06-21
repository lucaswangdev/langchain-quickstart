"""
第二课：SequentialChain - 顺序执行的链
=========================================
使用 LCEL 实现顺序执行

运行：python 02_sequential_chain.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例：中文 → 英文 → 解释（顺序执行）
# ============================================
print("【示例】翻译 + 解释 流水线")
print("-" * 30)

# Step 1：翻译
translate_template = PromptTemplate.from_template(
    "将以下中文翻译成英文：{chinese_text}"
)
translate_chain = translate_template | llm | StrOutputParser()

# Step 2：解释
explain_template = PromptTemplate.from_template(
    """请用简单的语言解释以下英文概念：
    
English: {english_text}

解释（中文）："""
)
explain_chain = explain_template | llm | StrOutputParser()

# 组合两个链（顺序执行）
full_chain = {"english_text": translate_chain} | explain_chain

# 运行
result = full_chain.invoke({
    "chinese_text": "人工智能是让机器具有人类智能的技术"
})

print("\n" + "=" * 30)
print(f"最终结果：\n{result}")
