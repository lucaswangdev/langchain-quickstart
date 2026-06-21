# 第二课：Chains - 组合 LLM 的力量

> 完成时间：1-2小时  
> 难度：⭐⭐

## 学习目标

- [ ] 理解什么是 Chain
- [ ] 学会使用 LLMChain
- [ ] 学会组合多个 Chain（SequentialChain）

## 2.1 什么是 Chain？

**Chain = 流水线**：把多个步骤串联起来，让数据像水一样流动。

```
输入 → Step 1 → Step 2 → Step 3 → 输出
            ↓        ↓        ↓
         处理1    处理2    处理3
```

### 为什么需要 Chain？

| 不用 Chain | 用 Chain |
|------------|----------|
| 所有逻辑写在一起，复杂难读 | 每个步骤独立，清晰易懂 |
| 改一个地方影响全局 | 改一个步骤，不影响其他 |
| 难以复用 | 步骤可重复使用 |

## 2.2 LLMChain：最基本的链

LLMChain = PromptTemplate + LLM + OutputParser

```bash
python 01_llm_chain.py
```

**代码解析：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. 创建 LLM
llm = ChatOpenAI()

# 2. 创建 Prompt 模板
template = PromptTemplate.from_template(
    "请用一句话解释 {concept}："
)

# 3. 创建 Chain
chain = LLMChain(llm=llm, prompt=template)

# 4. 运行 Chain
result = chain.invoke({"concept": "人工智能"})
print(result["text"])
```

## 2.3 SequentialChain：顺序执行

多个 Chain 按顺序执行，一个的输出是下一个的输入。

```bash
python 02_sequential_chain.py
```

**代码解析：**

```python
from langchain.chains import LLMChain, SequentialChain

# Chain 1：翻译
translate_chain = LLMChain(
    llm=llm,
    prompt=translate_template,
    output_key="english_text"
)

# Chain 2：总结（使用 Chain 1 的输出）
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_template,
    output_key="summary"
)

# 组合
full_chain = SequentialChain(
    chains=[translate_chain, summary_chain],
    input_variables=["chinese_text"],
    output_variables=["english_text", "summary"]
)

# 运行
result = full_chain.invoke({"chinese_text": "今天天气真好！"})
```

## 2.4 课程总结

### 你学到了什么？

- ✅ Chain 是组合 LLM 能力的流水线
- ✅ LLMChain = Prompt + LLM + Output
- ✅ SequentialChain 可以组合多个步骤
- ✅ Chain 让复杂流程清晰可维护

### 核心代码模式

```python
# 单个 Chain
chain = LLMChain(llm=llm, prompt=template)
result = chain.invoke({"输入": "值"})

# 顺序 Chain
full_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["x"],
    output_variables=["result"]
)
result = full_chain.invoke({"x": "输入值"})
```

## 下一步

- [ ] 继续 [03_agents/](../03_agents/) 学习 Agent
