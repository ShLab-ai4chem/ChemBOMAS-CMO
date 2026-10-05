from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import torch
from loguru import logger
import os, json
from core.settings import get_settings

settings = get_settings()

class RAGSystem:
    def __init__(self, docs_dir, model_path, embedding_model, faiss_index_path):
        self.docs_dir = docs_dir
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.faiss_index_path = faiss_index_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化组件
        self.vectorstore = self._get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = self._load_api_llm()

        logger.info("🤖 RAG 系统初始化完成")

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

    def create_chain(self):
        """创建新的对话链（包含独立的内存）"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
        )

    def run_queries(self, queries, chain=None):
        """
        执行一系列查询并返回最终结果
        :param queries: 查询列表（多轮对话）
        :param chain: 可选的现有对话链（用于保持对话历史）
        :return: 最终答案文本
        """
        # 创建新链（如果不提供）
        if chain is None:
            chain = self.create_chain()
            logger.info("🆕 创建新对话链")

        results = []
        for i, query in enumerate(queries):
            logger.info(f"\n🔍 查询 {i+1}/{len(queries)}: {query[:50]}...")
            result = chain.invoke({"question": query})
            results.append(result)

            # 记录来源文档
            for j, doc in enumerate(result["source_documents"]):
                file_name = os.path.basename(doc.metadata['source'])
                logger.info(f"{j+1}. {file_name} (Page {doc.metadata.get('page', 'N/A')})")

            logger.info("\n💡 生成的回答：")
            logger.info("="*100)
            logger.info(result["answer"])

        # 返回最后一个答案
        return results[-1]["answer"]