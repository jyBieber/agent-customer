# 企业级智能客服系统

> 基于 LangChain 1.0 + LangGraph 1.0 的智能客服系统（含流式输出）

## 项目简介

这是一个功能完整的企业级智能客服系统，使用最新的 LangChain 1.0 和 LangGraph 1.0 API 开发，集成了 RAG、实时流式通信、会话持久化等前沿技术。

### 核心特性

- ✅ **LangGraph Checkpointing** - 基于 `AsyncSqliteSaver` 的异步会话持久化，支持断线重连
- ✅ **RAG 最佳实践** - 使用 FAISS + BGE-small-zh 本地嵌入，余弦相似度检索
- ✅ **结构化输出** - 基于 Pydantic 的意图识别（含容错解析）
- ✅ **实时流式通信** - WebSocket + `astream_events` 逐字输出
- ✅ **完整业务流程** - 知识库检索、工单创建、人工转接
- ✅ **零成本部署** - 集成智谱 GLM-4-flash 免费模型 + 本地嵌入

### 技术栈

- **框架**: LangChain 1.3.9 + LangGraph 1.2.5
- **LLM**: 智谱 GLM-4-flash（通过 `langchain-openai` 兼容接口调用）
- **Embeddings**: HuggingFace BGE-small-zh-v1.5（本地运行）
- **向量数据库**: FAISS (CPU 版)
- **Web 框架**: FastAPI + WebSocket
- **数据库**: SQLite（业务库）+ aiosqlite（Checkpoint）
- **流式**: `astream_events` + 异步生成器

> **注**：本项目原计划使用 Qwen2.5-7B (SiliconFlow) 和官方 Middleware，但因兼容性问题已切换为智谱免费模型，并暂时注释了 Middleware 功能。

## 系统架构

```
┌─────────────────────────────────────────┐
│          前端（Web/命令行客户端）        │
└─────────────────────────────────────────┘
                  ↓ WebSocket (流式协议)
┌─────────────────────────────────────────┐
│      API Gateway（FastAPI）              │
│  - WebSocket 端点 /ws/{session_id}      │
│  - REST 端点 /stats                     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│     AI Agent（LangGraph + Checkpoint）   │
│  - create_react_agent（工具调用）        │
│  - 意图识别（Pydantic + 手动JSON解析）   │
│  - RAG 检索（search_knowledge 工具）     │
│  - 工单创建 / 人工转接                   │
│  - 流式输出（astream_events）           │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│           数据层                         │
│  - FAISS 索引（faiss_index/）            │
│  - SQLite：customer_service.db          │
│  - SQLite：checkpoints.db（异步）       │
└─────────────────────────────────────────┘
```

## 快速开始

### 环境准备

```bash
# 1. 克隆项目并进入目录
cd your-project

# 2. 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖（使用精简版 requirements.txt）
pip install -r requirements.txt

# 4. 配置环境变量
# 复制 .env.example 为 .env，填入您的智谱 API Key
ZHIPU_API_KEY=your_zhipu_api_key
```

### 启动服务

```bash
# 启动服务器（默认端口 8000）
python project10_customer_service_system_v2.py

# 另开终端，启动测试客户端
python project10_customer_service_system_v2.py client
```

## 使用示例

### 场景 1：知识库查询（流式输出）

```
👤 您: 退货政策是什么？

🤖 客服: （逐字显示）请问您需要退货的商品是什么？以及您希望退货的原因是什么呢？
```

系统会自动调用 `search_knowledge` 工具检索知识库，并结合上下文生成回答。

### 场景 2：创建工单

```
👤 您: 我的订单一直没收到，订单号 123456

🤖 客服: ✅ 工单已创建成功
工单编号：TK20250624143025
处理状态：待处理
预计响应时间：2小时内
```

### 场景 3：转接人工

```
👤 您: 我要投诉，转人工

🤖 客服: 🔄 正在为您转接人工客服...
转接原因：投诉
当前排队人数：3人
预计等待时间：3-5分钟
```

## 核心功能详解

### 1. 会话持久化（Checkpointing）

使用 `AsyncSqliteSaver` 实现异步会话状态存储，支持断线重连：

```python
# 在 CustomerServiceAgent.create 中异步创建
conn = await aiosqlite.connect("checkpoints.db")
checkpointer = AsyncSqliteSaver(conn)

# 每次对话传入 thread_id 自动保存/恢复
config = {"configurable": {"thread_id": session_id}}
result = await agent.ainvoke({"messages": messages}, config=config)
```

### 2. RAG 检索增强

- **文本分块**：`RecursiveCharacterTextSplitter`（chunk_size=500, overlap=50）
- **本地嵌入**：`HuggingFaceEmbeddings`（BAAI/bge-small-zh-v1.5）
- **向量存储**：FAISS 索引，持久化到 `faiss_index/`
- **检索工具**：`search_knowledge` 工具，Agent 自主决定何时调用

```python
@tool
def search_knowledge(query: str) -> str:
    docs = kb.search(query, k=2)  # Top-2 相似文档
    return 格式化后的文档内容
```

### 3. 意图识别（结构化输出）

使用 Pydantic 定义意图模型，并通过手动清理模型输出中的 Markdown 代码块实现容错解析：

```python
class Intent(BaseModel):
    type: str   # 退货/投诉/订单/退款/其他
    confidence: float
    topic: str
    urgency: int

# 直接调用模型，再正则提取 JSON
raw = model.invoke(prompt).content
json_str = re.search(r'\{.*\}', raw).group()
intent = Intent(**json.loads(json_str))
```

### 4. 流式输出

基于 `astream_events` 逐字推送，配合 WebSocket 实现打字机效果：

```python
async def process_stream(self, messages, session_id):
    async for event in self.agent.astream_events(
        {"messages": messages},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            yield event["data"]["chunk"].content
```

## 数据库结构

系统使用两个 SQLite 数据库：

- **customer_service.db**：业务数据
  - `users`：用户信息
  - `sessions`：会话记录
  - `messages`：消息历史
  - `tickets`：工单管理
  - `satisfaction`：满意度评价

- **checkpoints.db**：LangGraph 检查点（异步存储）

## 性能优化

1. **FAISS 索引持久化**：避免重复构建
2. **异步数据库驱动**：`aiosqlite` 支持并发
3. **流式输出**：减少首字延迟
4. **本地嵌入**：无需网络，响应更快

## 扩展建议

### 1. 启用 Middleware（需调整）
当前版本因兼容性注释了 `SummarizationMiddleware` 和 `ModelCallLimitMiddleware`，如需启用，请升级相关包并修改参数（`trigger` 和 `keep`）。

### 2. 增加用户认证
在 WebSocket 握手时验证 JWT Token。

### 3. 接入更多工具
可添加查订单、查物流等 API 工具，扩展 Agent 能力。

### 4. 前端集成
WebSocket 消息格式已标准化（`start`/`chunk`/`end`），可直接对接 Vue/React 页面。

## 注意事项

- 确保 `ZHIPU_API_KEY` 环境变量正确设置（智谱平台免费申请）。
- 首次启动会自动下载 BGE 模型（约 1.5 GB），请耐心等待。
- 若遇到 `sqlite3.IntegrityError`，已通过 `INSERT OR IGNORE` 处理。

