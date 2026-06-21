# 第四课：Memory - 记住对话历史

> 完成时间：1-2小时  
> 难度：⭐⭐

## 学习目标

- [ ] 理解为什么需要 Memory
- [ ] 学会使用 ConversationBufferMemory
- [ ] 理解不同类型的 Memory

## 4.1 什么是 Memory？

LLM 本身是**无状态**的 —— 每次调用都是独立的，不记得之前说过什么。

**Memory = 给 LLM 添加 "记忆"**

```
无 Memory：
  第1轮：你好 → 你好！
  第2轮：你叫什么？ → 我不知道你在问谁

有 Memory：
  第1轮：你好 → 你好！
  第2轮：你叫什么？ → 我叫 XXX，很高兴认识你！
```

## 4.2 Memory 类型

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| `BufferMemory` | 简单存储所有对话 | 简单聊天 |
| `BufferWindowMemory` | 只保留最近 N 轮 | 长对话节省 token |
| `EntityMemory` | 记住实体信息 | 需要理解实体 |
| `SummaryMemory` | 自动摘要 | 超长对话 |

## 4.3 对话 Memory

```bash
python 01_conversation_memory.py
```

**代码解析：**

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1. 创建 Memory
memory = ConversationBufferMemory()

# 2. 创建带 Memory 的 Chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 3. 对话
conversation.invoke("我叫小明")
conversation.invoke("我叫什么名字？")  # 会记住！
```

## 4.4 课程总结

### 你学到了什么？

- ✅ LLM 无状态，需要 Memory 来记住对话
- ✅ ConversationBufferMemory 最简单
- ✅ Memory 可以保存任意变量

### 核心代码模式

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# 对话
conversation.invoke("消息")
```

## 下一步

- [ ] 继续 [05_rag/](../05_rag/) 学习 RAG 实战
