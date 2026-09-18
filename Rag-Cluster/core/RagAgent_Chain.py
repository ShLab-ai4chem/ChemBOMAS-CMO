# ====================== Base ======================
from typing import Union
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import torch
from loguru import logger
import os, json, re
from core.settings import get_settings

# ====================== Agent ======================
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from agent_tools import *

settings = get_settings()
# Web search is opt-in and reads SERPER_API_KEY from the environment.

class EvaluateAgent:
    def __init__(self, llm):
        self.llm = llm
        self.evaluation_prompt = self._load_evaluation_prompt()

    def _load_evaluation_prompt(self):
        with open("core/eval_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    def evaluate(self, query: str, response: str) -> str:
        prompt = self.evaluation_prompt.format(query=query, response=response)

        result = self.llm.invoke(prompt).content.strip()

        pattern = r"(?i)evaluation\s*result\s*:\s*\[?\s*(sufficient|partially sufficient|insufficient)\s*\]?"
        match = re.search(pattern, result)

        logger.warning(f"Evaluation result: {result}")

        if match:
            return match.group(1).strip().lower()
        else:
            raise ValueError("Evaluation result not found in model output.")

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
        self.evaluator = EvaluateAgent(self.llm)

        # 定义Agent可以使用的工具列表
        self.tool_hierarchy  = self._create_tool_hierarchy()

        # 定义history, tool_output
        self.history = []
        self.tool_output = []

        logger.info("🤖 RAG 系统初始化完成")

    def _create_tool_hierarchy(self):
        """创建分层工具系统"""
        hierarchy = {
            "search_literature": create_retriever_tool(
                self.retriever,
                "search_literature", # 工具名称
                "Query references for answering user questions"
            ),
            "chem_database_search": chem_database_search,
        }

        if self.enable_web_search:
            hierarchy["web_search"] = google_search_tool

        return hierarchy

    def _execute_tool(self, tool, query: str) -> str:
        """执行单个工具并返回结果（支持多种工具类型）"""
        try:
            # 1. 优先处理 LangChain 标准工具接口
            if hasattr(tool, "run") and callable(tool.run):
                return tool.run(query)

            # 2. 处理 LangChain Runnable 接口（如新版本工具）
            elif hasattr(tool, "invoke") and callable(tool.invoke):
                try:
                    # 尝试两种调用方式以适应不同工具签名
                    try:
                        # 新版调用方式 (input 为字典)
                        result = tool.invoke({"input": query})
                    except TypeError:
                        # 旧版调用方式 (input 为字符串)
                        result = tool.invoke(query)

                    # 统一结果处理
                    if isinstance(result, dict):
                        return result.get("output", str(result))
                    return str(result)
                except Exception as invoke_e:
                    raise RuntimeError(f"Invoke call failed: {str(invoke_e)}") from invoke_e

            # 3. 处理函数式工具（包括 @tool 装饰器创建的工具）
            elif callable(tool):
                result = tool(query)
                return str(result) if not isinstance(result, str) else result

            # 4. 处理 StructuredTool 特殊类型
            elif hasattr(tool, "_run") and callable(tool._run):
                return tool._run(query)

            # 未知工具类型
            else:
                supported_types = [
                    "Tool(run)", "Runnable(invoke)",
                    "function", "StructuredTool(_run)"
                ]
                raise TypeError(
                    f"Unsupported tool type {type(tool)}. "
                    f"Supported types: {', '.join(supported_types)}"
                )

        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

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

    def _invoke_tool_with_synthesis_agent(self, tool, tool_name: str, query: str) -> str:
        """
        使用一个简单的“综合Agent”来调用工具并润色其输出。
        这个Agent负责将工具的原始输出转化为流畅的、针对性的回答。
        """
        logger.info(f"🧠 综合Agent正在使用工具 '{tool_name}'...")

        # 创建一个提示，让LLM基于工具输出进行综合
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a chemist and expert scientific assistant. "
            "You have access to use tools to find information about chemical substances. "
            "Your task is to read the provided tool output and deliver a concise, accurate answer to the user's original question. "
            "Do not mention that you used a tool, and do not add any information that is not present in the tool output. "
            "If the tool output lacks enough data to answer, clearly state that the information is insufficient."
            "Below is the conversation history:\n"
            "{history}\n\n"
            ),
            ("user",
            "Original Question:\n"
            "{input}\n\n"
            "Reference Information:\n"
            "{tool_output}\n\n"
            "Please provide a concise answer based solely on the tool output:"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        formatted_history = "\n".join(
            [f"{h['question']}\n {h['response']}" for h in self.history]
        )

        formatted_tool_output = "\n".join(
            [t['response'] for t in self.tool_output]
        )

        # 调用LLM进行综合
        agent = create_openai_tools_agent(self.llm, [tool], synthesis_prompt)

        agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)
        try:
            synthesized_response = agent_executor.invoke({
                "input": query,
                "history": formatted_history,
                "tool_output": formatted_tool_output
            })['output']
            logger.info(f"📝 综合后的回答:\n{synthesized_response}")
        except Exception as e:
            logger.warning(f"❌ 综合Agent调用失败: {tool_name}")
            synthesized_response = f"Error invoking tool {tool_name}: {str(e)}"

        return synthesized_response

    def run_stratified_query(self, query: Union[str, list[str]], history: list[dict] = None):
        """
        分层执行查询流程：
        1. 先使用RAG工具，由一个综合Agent调用并生成回答。
        2. 评估结果质量。
        3. 如果结果不充分，逐级尝试更高级的工具，每次都由综合Agent处理。
        """
        logger.info(f"🚀 开始分层查询")

        if history:
            self.history = history

        # 设定工具层级
        tool_levels = ["search_literature", "chem_database_search"]
        if self.enable_web_search:
            tool_levels.append("web_search")

        current_level_index = 0

        if isinstance(query, str):
            query = [query]

        for q_idx, q in enumerate(query):
            logger.info(f"🔍 查询内容: {q_idx+1}")
            self.tool_output = []  # 清空上一次的工具输出
            while current_level_index < len(tool_levels):
                current_level_name = tool_levels[current_level_index]
                logger.info(f"📌 第 {current_level_index + 1} 次检索, 当前层级: {current_level_name}")

                tool = self.tool_hierarchy.get(current_level_name)
                if not tool:
                    logger.warning(f"未在工具层级中找到工具: {current_level_name}")
                    current_level_index += 1
                    continue

                response = self._invoke_tool_with_synthesis_agent(tool, current_level_name, query)

                # 评估结果
                # evaluation = self.evaluator.evaluate(query, response)
                self.tool_output.append({
                    "level": current_level_name,
                    "response": response,
                    # "evaluation": evaluation
                })

                # 决策下一步
                # if evaluation == "sufficient":
                #     logger.success("✅ 结果充分，终止查询")
                #     break

                # logger.warning(f"🟡 结果被评估为 '{evaluation}'，升级到下一层级工具。")
                current_level_index += 1

            final_response = self.tool_output[-1]["response"]
            self.history.append({
                "question": q,
                "response": final_response
            })
        logger.info(f"\n✅ 查询结束。最终回答:\n{final_response}")

        return final_response, self.history
