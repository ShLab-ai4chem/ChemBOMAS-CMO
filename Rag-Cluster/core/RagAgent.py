# ====================== Base ======================
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import torch
from loguru import logger
import os, json
from core.settings import get_settings

# ====================== Agent ======================
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from agent_tools import *

settings = get_settings()
# Web search is opt-in and reads SERPER_API_KEY from the environment.

class RagAgent:
    def __init__(self,
                 docs_dir: str,
                 model_path: str,
                 embedding_model: str,
                 faiss_index_path: str,
                 enable_web_search: bool = False):
        """
        初始化RAG系统
        :param docs_dir: 文档目录
        :param model_path: 模型路径
        :param embedding_model: 嵌入模型
        :param faiss_index_path: FAISS索引路径
        :param enable_web_search: 是否启用网络搜索
        """
        self.docs_dir = docs_dir
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.faiss_index_path = faiss_index_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_web_search = enable_web_search

        # 初始化组件
        self.vectorstore = self._get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = self._load_api_llm()

        # 定义Agent可以使用的工具列表
        self.tools = self._create_tools()

        logger.info("🤖 RAG 系统初始化完成")

    def _create_tools(self):
        """创建Agent可以使用的工具列表"""
        retriever_tool = create_retriever_tool(
            self.retriever,
            "search_literature", # 工具名称
            "Query references for answering user questions"
        )

        tools = [retriever_tool, chem_database_search]  # 添加化学数据库查询工具

        if self.enable_web_search:

            tools.append(google_search_tool)
            logger.info("🌐 已启用联网搜索功能")

        return tools

    def _load_documents(self):
        loaders = [
            DirectoryLoader(
                self.docs_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            ),
            DirectoryLoader(
                self.docs_dir,
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                show_progress=True
            )
        ]
        return [doc for loader in loaders for doc in loader.load()]

    def _split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(docs)

    def _get_vectorstore(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device}
        )

        if os.path.exists(self.faiss_index_path):
            logger.info("⏳ 加载已有向量数据库...")
            return FAISS.load_local(
                self.faiss_index_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("⏳ 加载文档...")
            docs = self._load_documents()
            logger.info(f"✅ 已加载 {len(docs)} 个文档")

            logger.info("⏳ 分割文本...")
            splits = self._split_documents(docs)
            logger.info(f"📚 生成 {len(splits)} 个文本块")

            logger.info("⏳ 创建向量数据库...")
            vectorstore = FAISS.from_documents(splits, embedding_model)
            vectorstore.save_local(self.faiss_index_path)
            logger.info(f"🗄️ 向量数据库已保存至 {self.faiss_index_path}")
            return vectorstore

    def _load_api_llm(self):
        return ChatOpenAI(
            model=settings.model_name,
            temperature=0,
            base_url=settings.base_url,
            api_key=settings.api_key
        )

    def create_agent_executor(self):
        # Agent 需要一个提示模板，告诉它如何行动
        # MessagesPlaceholder 会被聊天记录和中间步骤填充

        with open("core/system_prompt.txt", "r", encoding="utf-8") as f:
            system = f.read()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 创建 Agent
        # 它将LLM、工具和提示组合在一起
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        # 创建 Agent Executor
        # 这是运行 Agent 的环境，负责调用工具并把结果返回给Agent
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True, # 设置为True可以看到Agent的思考过程
            handle_parsing_errors=True # 处理一些潜在的解析错误
        )

        return agent_executor

    def run_agent_query(self, query, chat_history=None):
        """
        使用Agent执行一个复杂的查询。
        :param query: 用户的查询字符串
        :param chat_history: 可选的对话历史
        :return: 包含最终答案和中间步骤的字典
        """
        if chat_history is None:
            chat_history = []

        agent_executor = self.create_agent_executor()
        logger.info(f"\n🚀 开始执行Agent查询: {query}")

        result = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })

        # 更新聊天记录
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=result["output"]),
            ]
        )

        logger.info("\n✅ Agent执行完成. 最终回答:")
        logger.info("="*100)
        logger.info(result["output"])

        return result, chat_history
