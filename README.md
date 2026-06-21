# LangChain Quickstart

> 从零开始，用实战带你入门 LangChain 开发

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green.svg)](https://python.langchain.com/)

## 这个项目是什么？

这是一个面向 **有 Python 基础、想快速上手 LangChain** 的学习者的实战项目。

## 什么是 LangChain？

### 一句话解释

**LangChain 是一个用于构建 LLM（大型语言模型）应用的开发框架**。

它就像 "乐高积木"，把和 LLM 交互的各种复杂操作封装成标准组件，让你不用从零开始造轮子。

### 它解决了什么问题？

| 痛点 | 没有 LangChain | 有 LangChain |
|------|----------------|--------------|
| 调用 LLM | 自己写 API 请求、处理响应 | `llm.invoke("你好")` 一行搞定 |
| 管理 Prompt | 字符串拼接，混乱难维护 | PromptTemplate 模板化管理 |
| 多轮对话 | 手动维护历史消息 | Memory 组件自动管理 |
| 连接外部工具 | 自己写 HTTP 请求 | Tools 组件，声明式定义 |
| 构建复杂流程 | 代码逻辑复杂难读 | Chain 链式组合，清晰直观 |

### 核心概念全景图

```
┌─────────────────────────────────────────────────────────┐
│                      LangChain                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│   │   LLM   │    │  Tool   │    │ Memory  │          │
│   │  大模型  │    │  工具   │    │  记忆   │          │
│   └────┬────┘    └────┬────┘    └────┬────┘          │
│        │              │              │                │
│        └──────────────┼──────────────┘                │
│                       ▼                                 │
│               ┌─────────────┐                          │
│               │    Chain    │                          │
│               │   链：组合   │                          │
│               │  它们工作   │                          │
│               └──────┬──────┘                          │
│                      │                                  │
│                      ▼                                  │
│               ┌─────────────┐                          │
│               │   Agent    │                          │
│               │  代理：    │                          │
│               │ 自动决策   │                          │
│               └─────────────┘                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 学习路径（循循渐进）

```
第一步：破冰 (30分钟)
├── 理解 LangChain 是什么
├── 环境配置
└── 完成第一个 LLM 调用

第二步：最少必要知识 (1小时)
├── Prompt Template
├── LLM Chain 基础
└── 简单聊天机器人

第三步：核心组件 (2-3小时)
├── Chains（链）- 组合 LLM 的力量
├── Agents（代理）- 让 LLM 自主决策
├── Memory（记忆）- 记住对话历史
└── Tools（工具）- 连接外部世界

第四步：实战应用 (2-3小时)
├── RAG（检索增强生成）
├── 本地文档问答
└── 构建自己的 AI 助手
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/langchain-quickstart.git
cd langchain-quickstart
```

### 2. 安装依赖（使用 uv）

```bash
# 如果没有安装 uv，先安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv && uv pip install langchain langchain-core langchain-openai python-dotenv

# 如果需要 RAG 或 Agent 功能，额外安装
uv pip install chromadb langchain-community duckduckgo-search
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```env
OPENAI_API_KEY=sk-你的密钥
```

> ⚠️ **安全提醒**：`.env` 已经在 `.gitignore` 中，不会被上传到 GitHub

### 4. 运行第一个例子

```bash
cd 01_quickstart
python 01_first_llm_call.py
```

看到输出 `Hello, World!` 即成功！

## 课程内容

| 课程 | 主题 | 难度 |
|------|------|------|
| [01_quickstart](./01_quickstart/) | 破冰：第一个 LLM 调用 | ⭐ |
| [02_chains](./02_chains/) | Chains：组合 LLM 的力量 | ⭐⭐ |
| [03_agents](./03_agents/) | Agents：让 LLM 自主决策 | ⭐⭐⭐ |
| [04_memory](./04_memory/) | Memory：记住对话历史 | ⭐⭐ |
| [05_rag](./05_rag/) | RAG：检索增强生成 | ⭐⭐⭐ |

## 项目结构

```
langchain-quickstart/
├── README.md                    # 你在这里！
├── .env.example                 # API Key 模板
├── pyproject.toml               # uv 项目配置
├── uv.lock                      # 依赖锁定文件（自动生成）
├── .gitignore
├── 01_quickstart/               # 第一课
│   ├── README.md
│   ├── 01_first_llm_call.py     # 第一行代码
│   └── 02_prompt_template.py    # Prompt 模板
├── 02_chains/
│   ├── README.md
│   ├── 01_llm_chain.py          # LLM Chain
│   └── 02_sequential_chain.py   # 顺序链
├── 03_agents/
│   ├── README.md
│   └── 01_basic_agent.py        # 基础 Agent
├── 04_memory/
│   ├── README.md
│   └── 01_conversation_memory.py
└── 05_rag/
    ├── README.md
    └── 01_simple_rag.py
```

## 常见问题

### Q: 需要什么基础？

- Python 基础（会写函数和类）
- 了解什么是 API（可选）
- 了解什么是 LLM（大语言模型）（可选）

### Q: 需要付费吗？

- 课程本身免费
- 运行代码需要 LLM API Key（如 OpenAI）
- 首次使用 OpenAI 有 $5 免费额度

### Q: 支持哪些 LLM？

LangChain 支持：OpenAI、Anthropic、Google、Azure、HuggingFace、本地模型等。本教程以 OpenAI 为例。

## 下一步

- [x] 完成 [01_quickstart](./01_quickstart/) 破冰课程
- [ ] 完成 [02_chains](./02_chains/) 学习 Chain
- [ ] 完成 [03_agents](./03_agents/) 学习 Agent
- [ ] 完成 [04_memory](./04_memory/) 学习 Memory
- [ ] 完成 [05_rag](./05_rag/) 实战 RAG

---

## 贡献

发现问题？欢迎提交 Issue 或 Pull Request！

## 许可证

MIT License
