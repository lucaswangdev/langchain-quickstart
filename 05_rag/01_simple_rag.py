"""
第五课：RAG - 检索增强生成
======================================
使用 ZVec 作为向量数据库

运行：python 01_simple_rag.py

安装依赖：
uv pip install zvec langchain langchain-core langchain-openai python-dotenv
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os
import zvec

root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=model_name)

# ============================================
# 示例：基于 ZVec + LangChain 的 RAG
# ============================================
print("=" * 60)
print("【RAG 示例】使用 ZVec 向量数据库")
print("=" * 60)

# 文档库
documents = [
    {"id": "doc_1", "content": "LangChain 是一个用于构建 LLM 应用的框架。", "topic": "langchain"},
    {"id": "doc_2", "content": "LangChain 的核心组件包括：Model、Prompt、Chain、Agent、Memory。", "topic": "核心组件"},
    {"id": "doc_3", "content": "Chain 用于组合多个组件形成工作流。", "topic": "chain"},
    {"id": "doc_4", "content": "Agent 让 LLM 能够自主决策和使用工具。", "topic": "agent"},
    {"id": "doc_5", "content": "Memory 用于保存对话历史。", "topic": "memory"},
    {"id": "doc_6", "content": "RAG 是检索增强生成，可以基于文档回答问题。", "topic": "rag"},
]

print(f"1. 文档库加载完成，共 {len(documents)} 个文档")

# ============================================
# 创建 ZVec 集合
# ============================================
print("\n2. 创建 ZVec 向量集合...")

db_path = "./zvec_db"

# 定义 Schema（4维向量）
schema = zvec.CollectionSchema(
    name="rag_docs",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# 尝试打开已存在的集合，如果不存在则创建
try:
    collection = zvec.open(path=db_path)
    print("   打开已有集合")
except:
    # 集合不存在，创建新的
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    collection = zvec.create_and_open(path=db_path, schema=schema)
    print("   创建新集合")

print("   ZVec 集合就绪！")

# ============================================
# 简化 Embedding（用于演示）
# ============================================
def simple_embed(text: str) -> list:
    """简化Embedding：用词频生成4维向量"""
    words = text.lower().split()
    vec = [0.0] * 4
    for word in set(words):
        idx = abs(hash(word)) % 4
        vec[idx] += 1.0
    # 归一化
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    else:
        vec = [0.25] * 4
    return vec

# ============================================
# 插入文档
# ============================================
print("\n3. 插入文档到向量数据库...")

docs_to_insert = []
for doc in documents:
    embedding = simple_embed(doc["content"])
    docs_to_insert.append(
        zvec.Doc(id=doc["id"], vectors={"embedding": embedding})
    )

collection.insert(docs_to_insert)
print(f"   已插入 {len(docs_to_insert)} 个文档")

# ============================================
# RAG 查询函数
# ============================================
rag_template = PromptTemplate.from_template(
    """基于以下上下文回答问题。

上下文：
{context}

问题：{question}

回答："""
)

def rag_query(question: str, top_k: int = 2):
    """RAG 查询"""
    # 1. 将问题转为向量
    question_vec = simple_embed(question)
    
    # 2. 向量搜索
    results = collection.query(
        zvec.Query("embedding", vector=question_vec),
        topk=top_k
    )
    
    # 3. 获取相关文档
    doc_map = {d["id"]: d for d in documents}
    relevant_docs = []
    for result in results:
        if result.id in doc_map:
            relevant_docs.append(doc_map[result.id])
    
    # 4. 构建上下文
    context = "\n".join([d["content"] for d in relevant_docs])
    
    # 5. LLM 生成
    prompt = rag_template.invoke({"context": context, "question": question})
    response = llm.invoke(prompt)
    
    return response.content, relevant_docs

print("4. RAG 系统就绪！")

# ============================================
# 问答
# ============================================
print("\n" + "=" * 60)
print("开始问答（输入 'quit' 退出）")
print("=" * 60)

while True:
    question = input("\n问题：")
    if question.lower() == "quit":
        break
    
    if not question.strip():
        continue
    
    answer, docs = rag_query(question)
    
    print(f"\n答案：{answer}")
    print("-" * 60)
    print("参考文档：")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. [{doc['topic']}] {doc['content']}")

print("\n清理 ZVec 数据库...")
# collection.close()  # ZVec不需要手动关闭
