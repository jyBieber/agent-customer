"""
综合实战：企业级智能客服系统（LangChain 1.0 + LangGraph 1.0）

最新特性：
- LangGraph Checkpointing（会话持久化）
- 官方 Middleware（SummarizationMiddleware）
- RAG 最佳实践（RecursiveCharacterTextSplitter + FAISS）
- 结构化输出（意图识别）
- WebSocket 实时通信
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import uuid

from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph 1.0 Checkpointing
from langgraph.checkpoint.sqlite import SqliteSaver

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

#流式输出，分布式
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# load_dotenv()
from pathlib import Path
env_path = Path(__file__).parent / ".env.example"
load_dotenv(dotenv_path=env_path, override=True)

# ==================== 数据模型 ====================

class Intent(BaseModel):
    """用户意图（结构化输出）"""
    type: str = Field(
        description="意图类型：inquiry（咨询）/complaint（投诉）/order（订单）/refund（退款）/other（其他）"
    )
    confidence: float = Field(description="置信度0-1", ge=0, le=1)
    topic: str = Field(description="具体主题")
    urgency: int = Field(description="紧急程度1-5", ge=1, le=5)

class Ticket(BaseModel):
    """工单"""
    ticket_id: str
    user_id: str
    title: str
    description: str
    status: str = "pending"
    priority: int = 1

# ==================== 数据库管理 ====================

class Database:
    """数据库管理"""

    def __init__(self, db_path: str = "customer_service.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建表
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            vip_level INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS satisfaction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()

    def create_user(self, user_id: str, name: str, phone: str = None):
        """创建用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR IGNORE INTO users (user_id, name, phone) VALUES (?, ?, ?)",
            (user_id, name, phone)
        )

        conn.commit()
        conn.close()


    def create_session(self, session_id: str, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 检查是否存在
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id)
            )
        else:
            # 可选项：更新状态或时间
            cursor.execute(
                "UPDATE sessions SET status = 'active', closed_at = NULL WHERE session_id = ?",
                (session_id,)
            )
        conn.commit()
        conn.close()

    def save_message(self, session_id: str, role: str, content: str):
        """保存消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )

        conn.commit()
        conn.close()

    def create_ticket(self, ticket: Ticket):
        """创建工单"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tickets (ticket_id, user_id, session_id, title, description, priority)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticket.ticket_id, ticket.user_id, None, ticket.title, ticket.description, ticket.priority))

        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict:
        """获取统计数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 总会话数
        cursor.execute("SELECT COUNT(*) FROM sessions")
        stats["total_sessions"] = cursor.fetchone()[0]

        # 总消息数
        cursor.execute("SELECT COUNT(*) FROM messages")
        stats["total_messages"] = cursor.fetchone()[0]

        # 总工单数
        cursor.execute("SELECT COUNT(*) FROM tickets")
        stats["total_tickets"] = cursor.fetchone()[0]

        # 平均满意度
        cursor.execute("SELECT AVG(rating) FROM satisfaction")
        avg_rating = cursor.fetchone()[0]
        stats["avg_satisfaction"] = round(avg_rating, 2) if avg_rating else 0

        conn.close()
        return stats

# ==================== 知识库（RAG 最佳实践）====================

class KnowledgeBase:
    """知识库（使用 2025 RAG 最佳实践）"""

    def __init__(self, persist_path: str = "faiss_index"):
        self.persist_path = persist_path


        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},  # 如有GPU可改为 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 文本分割器（RecursiveCharacterTextSplitter）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # 适中的块大小
            chunk_overlap=50,    # 10% 重叠
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        # 3. 初始化或加载向量库
        if os.path.exists(persist_path):
            print(f"📂 加载已有向量库：{persist_path}")
            self.vectorstore = FAISS.load_local(
                persist_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"🆕 创建新向量库")
            self.docs = self._create_knowledge_docs()
            self.vectorstore = FAISS.from_documents(
                self.docs,
                self.embeddings
            )
            # 持久化
            self.vectorstore.save_local(persist_path)
            print(f"💾 向量库已保存：{persist_path}")

    def _create_knowledge_docs(self) -> List[Document]:
        """创建知识库文档"""
        knowledge = [
            {
                "content": """退货政策详细说明：

我们提供7天无理由退货服务。具体规则如下：
1. 商品需保持完好，包装完整
2. 不影响二次销售
3. 附带完整的发票和配件
4. 特殊商品（如生鲜、定制品）不支持退货
5. 退货运费：非质量问题由买家承担，质量问题由卖家承担
                """,
                "category": "退货"
            },
            {
                "content": """发票申请流程：

如何申请发票：
1. 登录您的账户
2. 进入"我的订单"页面
3. 找到对应订单，点击"申请发票"
4. 选择发票类型（电子发票/纸质发票）
5. 填写抬头信息（个人/企业）
6. 提交申请

电子发票将在1个工作日内发送至您的邮箱。
纸质发票将随商品一起寄出。
                """,
                "category": "发票"
            },
            {
                "content": """配送说明与时效：

全国包邮政策：
- 普通地区：订单满99元包邮，3-5个工作日送达
- 偏远地区：订单满199元包邮，5-7个工作日送达
- VIP会员：全部订单包邮，优先处理

配送时段：
- 工作日：9:00-18:00
- 周末：10:00-17:00

支持预约配送时间。
                """,
                "category": "配送"
            },
            {
                "content": """支付方式与优惠：

支持以下支付方式：
1. 微信支付（支持花呗分期）
2. 支付宝（支持信用卡）
3. 银行卡支付（储蓄卡/信用卡）
4. 企业转账（需联系客服）

支付优惠：
- 微信支付：每周三随机立减
- 支付宝：每月5/15/25号减10元
- 信用卡：部分银行支持免息分期
                """,
                "category": "支付"
            },
            {
                "content": """会员等级体系：

会员分为四个等级：
1. 普通会员：注册即可获得
2. 银卡会员：累计消费满1000元
3. 金卡会员：累计消费满5000元
4. 钻石会员：累计消费满10000元

会员权益：
- 银卡：95折优惠 + 生日礼包
- 金卡：9折优惠 + 专属客服 + 优先发货
- 钻石：85折优惠 + 专属客服 + 免费上门取退 + 年度大礼
                """,
                "category": "会员"
            },
            {
                "content": """积分规则与兑换：

积分获取：
- 每消费1元获得1积分
- 完成评价额外获得10积分
- 推荐好友注册获得50积分
- 每日签到获得5积分

积分用途：
- 100积分 = 1元抵扣券
- 500积分可兑换包邮券
- 1000积分可兑换优惠券礼包
- 5000积分可兑换实物礼品

积分有效期：2年
                """,
                "category": "积分"
            },
            {
                "content": """联系客服渠道：

多种联系方式：
1. 在线客服：工作时间 9:00-22:00
2. 客服热线：400-123-4567（工作时间 9:00-18:00）
3. 客服邮箱：service@example.com
4. 微信客服：添加微信号 CS-12345
5. 官方微博：@品牌官方客服

VIP会员享有专属客服，24小时响应。
                """,
                "category": "客服"
            },
            {
                "content": """售后服务政策：

质保承诺：
- 所有商品提供30天质保
- 非人为损坏免费维修
- 质量问题7天内可换货
- 严重质量问题可申请退款

售后流程：
1. 联系客服说明问题
2. 提供订单号和问题照片
3. 客服审核并提供解决方案
4. 寄回商品（质量问题运费由卖家承担）
5. 收到商品后3个工作日内处理完成
                """,
                "category": "售后"
            }
        ]

        docs = []
        for item in knowledge:
            # 使用 text_splitter 分块
            chunks = self.text_splitter.create_documents(
                texts=[item["content"]],
                metadatas=[{"category": item["category"]}]
            )
            docs.extend(chunks)

        print(f"📄 知识库文档数：{len(docs)}")
        return docs

    def search(self, query: str, k: int = 3) -> List[Document]:
        """检索知识库（余弦相似度）"""
        return self.vectorstore.similarity_search(query, k=k)

    def add_document(self, content: str, metadata: dict):
        """动态添加文档"""
        chunks = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[metadata]
        )
        self.vectorstore.add_documents(chunks)
        # 重新保存
        self.vectorstore.save_local(self.persist_path)

# ==================== 工具定义 ====================

kb = KnowledgeBase()
db = Database()

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库

    Args:
        query: 搜索关键词
    """
    docs = kb.search(query, k=2)
    if docs:
        results = []
        for doc in docs:
            category = doc.metadata.get("category", "未分类")
            results.append(f"【{category}】{doc.page_content}")
        return "\n\n---\n\n".join(results)
    return "未找到相关信息，建议转接人工客服。"

@tool
def create_ticket_tool(title: str, description: str, user_id: str) -> str:
    """创建工单

    Args:
        title: 工单标题
        description: 问题描述
        user_id: 用户ID
    """
    ticket_id = f"TK{datetime.now().strftime('%Y%m%d%H%M%S')}"
    ticket = Ticket(
        ticket_id=ticket_id,
        user_id=user_id,
        title=title,
        description=description,
        priority=3
    )

    db.create_ticket(ticket)

    return f"""✅ 工单已创建成功

工单编号：{ticket_id}
处理状态：待处理
预计响应时间：2小时内

我们会尽快处理您的问题，请保持联系方式畅通。您可以通过工单号查询进度。"""

@tool
def transfer_to_human(reason: str) -> str:
    """转接人工客服

    Args:
        reason: 转接原因
    """
    return f"""🔄 正在为您转接人工客服...

转接原因：{reason}
当前排队人数：3人
预计等待时间：3-5分钟

请稍候，我们的人工客服将尽快为您服务。"""

def identify_intent(model, user_message: str) -> Intent:
    # 1. 使用更明确的指令，并设置 response_format 为 json_object（如果模型支持）
    prompt = f"""分析用户消息的意图。**你的回复必须是一个有效的 JSON 对象，不要包含任何其他文字、解释或 Markdown 代码块。**

用户消息：{user_message}

JSON 格式必须严格如下：
{{"type": "意图类型", "confidence": 置信度(0-1小数), "topic": "具体主题", "urgency": 紧急程度(1-5整数)}}
"""
    # 2. 直接调用模型获取原始字符串，而不使用 with_structured_output
    raw_response = model.invoke(prompt).content
    print(f"模型原始返回: {raw_response}")  # 调试用，可注释掉

    # 3. 清理响应，提取 JSON
    import re
    # 移除 Markdown 代码块标记
    cleaned = re.sub(r'```json\s*', '', raw_response)
    cleaned = re.sub(r'```\s*', '', cleaned)
    # 尝试查找 JSON 对象
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            # 4. 手动解析并校验
            data = json.loads(json_str)
            return Intent(
                type=data["type"],
                confidence=float(data["confidence"]),
                topic=data["topic"],
                urgency=int(data["urgency"])
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"JSON 解析失败: {e}")
    # 如果解析失败，返回一个默认意图
    return Intent(type="other", confidence=0.5, topic="未知", urgency=3)

# ==================== 智能客服 Agent ====================

class CustomerServiceAgent:
    """智能客服 Agent（LangChain 1.0 + Middleware + Checkpointing）"""

    def __init__(self, model, tools, system_prompt, checkpointer, use_middleware=False):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

        from langgraph.prebuilt import create_react_agent

        # 将原本的 self.agent = create_agent(...) 替换为：
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.checkpointer
        )

    @classmethod
    async def create(cls, use_middleware=True):
        # 异步创建 checkpointer 和模型（与之前相同）
        conn = await aiosqlite.connect("checkpoints.db")
        checkpointer = AsyncSqliteSaver(conn)
        model = ChatOpenAI(
            model="glm-4-flash",
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            temperature=0.3,
            streaming=True  # 👈 关键
        )
        tools = [search_knowledge, create_ticket_tool, transfer_to_human]

        # 4. 系统提示（保持原样）
        system_prompt = """你是一个专业、友好的智能客服助手。

        核心职责：
        1. 礼貌、耐心地回答客户问题
        2. 使用 search_knowledge 搜索知识库获取准确信息
        3. 无法解决或复杂问题使用 create_ticket_tool 创建工单
        4. 客户情绪激动或投诉时使用 transfer_to_human 转接人工

        重要原则：
        - 始终保持专业和耐心
        - 不确定时明确告知用户，不要编造信息
        - 优先使用知识库回答
        - 问题解决后询问是否还有其他需要帮助的

        沟通风格：
        - 称呼客户为"您"
        - 使用礼貌用语（"请问"、"感谢"、"抱歉"）
        - 避免使用专业术语，用简单易懂的语言
        """

        # 直接调用 __init__ 创建实例（注意 __init__ 是同步的，但这里在异步函数中调用没问题）
        instance = cls(model, tools, system_prompt, checkpointer, use_middleware=False)
        return instance

    def process(self, messages: list, session_id: str) -> str:
        """处理对话（带会话持久化）"""

        # 配置（指定 thread_id 实现会话持久化）
        config = {"configurable": {"thread_id": session_id}}

        # 调用 agent
        result = self.agent.invoke(
            {"messages": messages},
            config=config  # 👈 传入配置
        )

        # 提取最后的 AI 回复
        for msg in reversed(result["messages"]):
            if msg.type == "ai" and msg.content:
                return msg.content

        return "抱歉，我暂时无法回答这个问题。"

    #流式输出
    async def process_stream(self, messages: list, session_id: str):
        config = {"configurable": {"thread_id": session_id}}
        print("🔄 开始流式处理...")  # 1. 进入方法
        try:
            async for event in self.agent.astream_events(
                    {"messages": messages},
                    config=config,
                    version="v2"
            ):
                print(f"📨 收到事件: {event['event']}")  # 2. 打印所有事件
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(f"📝 产出 chunk: {chunk.content}")  # 3. 打印每个 chunk
                        yield chunk.content
                elif event["event"] == "on_tool_start":
                    yield f"\n[🔧 正在调用工具: {event['name']}]\n"
        except Exception as e:
            print(f"❌ process_stream 异常: {e}")  # 4. 捕获异常
            import traceback
            traceback.print_exc()
            yield f"抱歉，处理您的请求时出现错误：{e}"

# ==================== FastAPI 服务 ====================

app = FastAPI(
    title="智能客服系统（LangChain 1.0）",
    version="2.0.0",
    description="基于 LangChain 1.0 + LangGraph 最新官方 API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
agent = None

@app.on_event("startup")
async def startup():
    """启动"""
    global agent
    print("🚀 初始化智能客服系统（LangChain 1.0）...")
    # agent = CustomerServiceAgent()
    agent = await CustomerServiceAgent.create()  # 异步创建
    print("✅ 系统启动完成")

@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "智能客服系统",
        "version": "2.0.0",
        "framework": "LangChain 1.0 + LangGraph 1.0",
        "features": [
            "LangGraph Checkpointing（会话持久化）",
            "官方 Middleware（对话总结）",
            "RAG 最佳实践（FAISS + 余弦相似度）",
            "结构化输出（意图识别）"
        ],
        "status": "running"
    }

@app.get("/stats")
async def get_stats():
    """获取统计"""
    return db.get_statistics()

# WebSocket 连接管理
class ConnectionManager:
    """连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """断开"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        """发送消息"""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 端点"""
    await manager.connect(websocket, session_id)

    # 创建会话
    user_id = f"user_{session_id[:8]}"
    db.create_user(user_id, f"用户{session_id[:8]}")
    db.create_session(session_id, user_id)

    # 欢迎消息
    welcome = {
        "role": "assistant",
        "content": """您好！我是智能客服助手 🤖

我可以帮您：
✅ 查询退货、发票、配送等政策
✅ 解答会员、积分相关问题
✅ 创建工单并跟进处理
✅ 转接人工客服

有什么可以帮您？""",
        "timestamp": datetime.now().isoformat()
    }
    await manager.send_message(session_id, welcome)

    # 对话历史（LangGraph Checkpointing 会自动管理）
    messages = []

    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            # 保存用户消息
            db.save_message(session_id, "user", user_message)
            messages.append({"role": "user", "content": user_message})

            # 意图识别（可选）
            try:
                intent = identify_intent(agent.model, user_message)
                print(f"📊 意图识别：{intent.type}（{intent.confidence:.0%}）- {intent.topic}")
            except Exception as e:
                print(f"⚠️  意图识别失败：{e}")

            # 处理消息（流式）
            full_response = ""
            # 先发送一个"开始"信号（可选）
            await manager.send_message(session_id, {"type": "start"})

            async for chunk in agent.process_stream(messages, session_id):
                full_response += chunk
                # 逐块发送给前端
                await manager.send_message(session_id, {
                    "type": "chunk",
                    "content": chunk
                })

            # 流式传输结束，保存完整消息
            db.save_message(session_id, "assistant", full_response)
            messages.append({"role": "assistant", "content": full_response})

            # 发送结束信号
            await manager.send_message(session_id, {
                "type": "end",
                "full_content": full_response
            })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"会话 {session_id} 已断开")

# ==================== 测试客户端 ====================

def test_client():
    """测试客户端（命令行版）"""
    import websocket
    import json
    import threading

    session_id = f"test_{uuid.uuid4().hex[:8]}"
    ws_url = f"ws://localhost:8000/ws/{session_id}"

    print("="*70)
    print("🤖 智能客服系统（LangChain 1.0 + LangGraph）")
    print("="*70)
    print(f"会话ID: {session_id}\n")

    def on_message(ws, message):
        data = json.loads(message)
        if data.get("type") == "chunk":
            # 流式片段，追加到当前行（示例简单打印，无换行）
            print(data["content"], end="", flush=True)
        elif data.get("type") == "end":
            print("\n")  # 换行，结束回复
        elif data.get("type") == "start":
            pass  # 可忽略
        else:
            # 兼容旧格式（如有）
            if "content" in data:
                print(f"\n🤖 客服: {data['content']}\n")

    def on_error(ws, error):
        print(f"❌ 错误: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("\n👋 连接已关闭")

    def on_open(ws):
        """连接打开"""
        print("✅ 已连接到客服系统\n")

        def run():
            while True:
                user_input = input("👤 您: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    ws.close()
                    break

                ws.send(json.dumps({"message": user_input}))

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    ws.run_forever()

# ==================== 主程序 ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "client":
        # 运行测试客户端
        test_client()
    else:
        # 运行服务器
        import uvicorn

        print("="*70)
        print("🚀 启动企业级智能客服系统（LangChain 1.0 + LangGraph）")
        print("="*70)
        print("\n✨ 最新特性:")
        print("  - LangGraph Checkpointing（会话持久化）")
        print("  - 官方 Middleware（对话总结、调用限制）")
        print("  - RAG 最佳实践（FAISS + 余弦相似度）")
        print("  - 结构化输出（意图识别）")
        print("\n📡 访问:")
        print("  - API 文档: http://localhost:8000/docs")
        print("  - 统计数据: http://localhost:8000/stats")
        print("  - WebSocket: ws://localhost:8000/ws/{session_id}")
        print("\n🧪 测试客户端:")
        print("  python project10_customer_service_system_v2.py client\n")

        uvicorn.run(app, host="0.0.0.0", port=8000)
