# 第三课：Agents - 让 LLM 自主决策

> 完成时间：2-3小时  
> 难度：⭐⭐⭐

## 学习目标

- [ ] 理解什么是 Agent
- [ ] 学会使用基础 Agent
- [ ] 理解 Tool 的概念

## 3.1 什么是 Agent？

**Agent = LLM + Tools + 推理引擎**

Agent 会思考：
1. 我需要做什么？
2. 我应该使用哪个工具？
3. 工具返回了什么？
4. 我现在知道答案了吗？还需要继续吗？

```
Agent 思考过程：
┌─────────────────────────────────────┐
│  🔍 思考：我该怎么做？              │
│     ↓                               │
│  🛠️ 行动：使用 Tool X               │
│     ↓                               │
│  👁️ 观察：Tool 返回了什么？         │
│     ↓                               │
│  🔄 循环：直到得到答案              │
└─────────────────────────────────────┘
```

## 3.2 核心概念

### Tool（工具）

Tool 是 Agent 可以调用的外部能力：
- `search` - 搜索信息
- `calculator` - 数学计算
- `wikipedia` - 查 Wikipedia
- 或者你自己定义的任何函数

### Agent 类型

| Agent | 特点 | 适用场景 |
|-------|------|----------|
| `zero-shot-react` | 根据描述选择工具 | 通用场景 |
| `conversational-react` | 对话式，更自然 | 聊天机器人 |
| `react-docstore` | 连接文档数据库 | 问答系统 |

## 3.3 第一个 Agent

```bash
python 01_basic_agent.py
```

**代码解析：**

```python
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun

# 1. 创建 LLM
llm = ChatOpenAI()

# 2. 定义 Tools
tools = [
    DuckDuckGoSearchRun(),  # 搜索工具
    WikipediaQueryRun(),     # Wikipedia 工具
]

# 3. 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 显示思考过程
)

# 4. 运行 Agent
result = agent.invoke("今天北京的天气怎么样？")
```

## 3.4 课程总结

### 你学到了什么？

- ✅ Agent = LLM + Tools + 推理
- ✅ Tool 是 Agent 的外部能力
- ✅ Agent 会自主决定使用哪个工具
- ✅ 循环思考直到得到答案

### 核心代码模式

```python
from langchain.agents import AgentType, initialize_agent

agent = initialize_agent(
    tools=[tool1, tool2, ...],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.invoke("你的问题")
```

## 下一步

- [ ] 继续 [04_memory/](../04_memory/) 学习记忆组件
