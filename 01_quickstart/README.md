# 第一课：破冰 - 第一个 LLM 调用

> 完成时间：30分钟  
> 难度：⭐

## 学习目标

- [x] 理解 LangChain 是什么
- [x] 配置开发环境
- [x] 完成第一次 LLM 调用

## 1.1 什么是 LangChain？

### 一句话解释

**LangChain 是一个用于构建 LLM 应用的开发框架**。

想象你要和一个很聪明但健忘的助手工作：
- 你需要告诉他 "你是谁"（System Prompt）
- 你需要给他 "工具" 来完成任务（Tools）
- 你需要帮他 "记忆" 对话内容（Memory）
- 你需要把多个步骤 "链" 在一起（Chain）

LangChain 就是帮你管理这些的框架。

### 为什么需要 LangChain？

| 不用框架 | 用 LangChain |
|----------|--------------|
| 自己处理 API 调用、错误重试 | 一行代码调用 LLM |
| 字符串拼接 Prompt，易错难维护 | PromptTemplate 模板化管理 |
| 手动管理对话历史 | Memory 组件自动处理 |
| 自己实现工具调用逻辑 | 声明式定义 Tools |
| 代码逻辑纠缠在一起 | Chain 链式组合，清晰易懂 |

## 1.2 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-key-here
```

> ⚠️ **重要**：`.env` 不会被提交到 GitHub（已在 .gitignore 中）

### 3. 验证环境

```bash
python -c "from langchain_openai import ChatOpenAI; print('OK')"
```

看到 `OK` 即成功！

## 1.3 第一个 LLM 调用

这是你使用 LangChain 的第一行代码：

```bash
python 01_first_llm_call.py
```

**发生了什么？**

1. LangChain 连接 OpenAI API
2. 发送你的问题
3. 打印返回的答案

**代码解析：**

```python
from langchain_openai import ChatOpenAI  # 导入 LangChain 的 OpenAI 客户端

llm = ChatOpenAI()                      # 创建 LLM 实例
response = llm.invoke("Hello, World!") # 调用 LLM
print(response.content)                 # 打印回复
```

就这么简单！

## 1.4 Prompt 模板

在实际应用中，我们通常不会直接写死 Prompt，而是使用 **PromptTemplate** 来管理：

```bash
python 02_prompt_template.py
```

**代码解析：**

```python
from langchain_core.prompts import PromptTemplate

# 定义一个模板
template = PromptTemplate.from_template(
    "请把以下句子翻译成 {language}：{sentence}"
)

# 用模板生成最终 Prompt
prompt = template.invoke({
    "language": "英文",
    "sentence": "你好，世界！"
})

# 发送给 LLM
response = llm.invoke(prompt)
print(response.content)
```

**输出：**
```
Hello, World!
```

## 1.5 课程总结

### 你学到了什么？

- ✅ LangChain 是一个 LLM 应用开发框架
- ✅ 使用 `ChatOpenAI` 调用 OpenAI 模型
- ✅ 使用 `PromptTemplate` 管理 Prompt
- ✅ 环境变量管理 API Key（安全！）

### 核心代码模式

```python
# 1. 创建 LLM 实例
llm = ChatOpenAI()

# 2. 调用
response = llm.invoke("你的问题")

# 3. 获取结果
print(response.content)
```

## 下一步

- [ ] 继续 [02_chains/](../02_chains/) 学习 Chain
- [ ] 理解如何组合多个步骤

---

**问题？** 欢迎提交 Issue！
