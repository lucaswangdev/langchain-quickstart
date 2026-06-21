# 第五课：RAG - 检索增强生成

> 完成时间：2-3小时  
> 难度：⭐⭐⭐

## 学习目标

- [ ] 理解什么是 RAG
- [ ] 理解 Vector Database 的作用
- [ ] 实现简单的 RAG 问答系统

## 5.1 什么是 RAG？

**RAG = Retrieval + Augmentation + Generation**

```
                    ┌─────────────┐
                    │   文档库    │
                    └──────┬──────┘
                           │ 检索
                           ▼
用户问题 ──→ 增强 ──→ LLM ──→ 答案
              ↑
         相关文档片段
```

### 为什么需要 RAG？

| 不用 RAG | 用 RAG |
|----------|--------|
| LLM 可能 "胡编乱造" | 基于真实文档回答 |
| 知识受限于训练数据 | 可访问最新信息 |
| 无法追溯答案来源 | 可以引用原文 |

## 5.2 RAG 工作流程

```
1. 文档入库
   文档 → 分块 → Embedding → 向量数据库

2. 问答
   问题 → Embedding → 相似度搜索 → 找到相关片段 → LLM 生成答案
```

## 5.3 简单的 RAG 实现

```bash
python 01_simple_rag.py
```

**代码解析：**

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# 1. 加载文档
loader = TextLoader("文档.txt")
documents = loader.load()

# 2. 分块
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, embeddings)

# 4. 相似度搜索
docs = db.similarity_search("相关问题")

# 5. 生成答案
llm = ChatOpenAI()
# ... 结合文档和问题，让 LLM 回答
```

## 5.4 课程总结

### 你学到了什么？

- ✅ RAG = 检索 + 增强 + 生成
- ✅ Embedding 把文本变成向量
- ✅ 向量数据库支持相似度搜索
- ✅ RAG 让 LLM "阅读" 自己的文档

### 核心代码模式

```python
# 1. 建库
vectorstore = Chroma.from_documents(documents, embeddings)

# 2. 搜索
docs = vectorstore.similarity_search(question)

# 3. 生成
answer = llm.invoke(context + question)
```

## 后续学习

恭喜完成基础课程！接下来可以学习：

- [ ] 使用 FAISS 作为向量数据库（支持本地）
- [ ] 实现多文档问答
- [ ] 添加 Memory 实现多轮 RAG 对话
- [ ] 使用 LangServe 部署为 API
