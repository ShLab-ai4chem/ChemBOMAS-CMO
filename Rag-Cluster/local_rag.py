from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from loguru import logger
import os, json
from core.paths import DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_PATH, DEFAULT_LOCAL_MODEL, configured_path
# import nltk
# nltk.download('averaged_perceptron_tagger')

# 配置参数
DOCS_DIR = configured_path("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)  # 文档目录
MODEL_PATH = configured_path("RAG_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)  # 本地模型路径
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL  # 小型嵌入模型
FAISS_INDEX_PATH = configured_path("RAG_INDEX_PATH", DEFAULT_INDEX_PATH)  # 向量数据库保存路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# 1. 加载文档
def load_documents():
    loaders = [
        DirectoryLoader(
            DOCS_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        ),
        DirectoryLoader(
            DOCS_DIR,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )
    ]
    return [doc for loader in loaders for doc in loader.load()]

# 2. 分割文本
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

# 3. 创建或加载向量存储
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE}
    )

    # 检查是否已有保存的向量数据库
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("⏳ 加载已有向量数据库...")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("⏳ 加载文档...")
        docs = load_documents()
        logger.info(f"✅ 已加载 {len(docs)} 个文档")

        logger.info("⏳ 分割文本...")
        splits = split_documents(docs)
        logger.info(f"📚 生成 {len(splits)} 个文本块")

        logger.info("⏳ 创建向量数据库...")
        vectorstore = FAISS.from_documents(splits, embedding_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info(f"🗄️ 向量数据库已保存至 {FAISS_INDEX_PATH}")
        return vectorstore

# 4. 加载本地模型
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)

# 5. 创建RAG系统
def create_rag_system():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info("⏳ 加载本地模型...")
    llm = load_local_llm()
    logger.info(f"🤖 模型 {MODEL_PATH} 加载完成")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# 6. 运行查询
def run_query(qa_system, query):
    logger.info("\n🔍 检索文档")
    logger.info("!"*50)
    result = qa_system.invoke({"query": query})

    logger.info("\n🔍 检索到的来源：")
    for i, doc in enumerate(result["source_documents"]):
        # 提取文件名（不带路径）
        file_name = os.path.basename(doc.metadata['source'])
        logger.info(f"{i+1}. {file_name} (Page {doc.metadata.get('page', 'N/A')})")

    logger.info("\n💡 生成的回答：")
    logger.info(result["result"])

# 主程序
if __name__ == "__main__":
    rag_system = create_rag_system()

    logger.info("\n" + "="*50)
    logger.info("RAG 系统准备就绪！")
    logger.info("="*50)
    json_string = json.load(open("./json_files/suzuki/dry_sum_suzuki.json", "r", encoding='utf-8'))['base']
    query = f"""
Please operate on the following json file based on the content in the document:
1. The content of json file is {json_string}
2. You need to cluster chemical substances, for instance, solvents can be grouped according to their polarity
3. Now, you only need to complete the clustering of the bases
4. Please note: The classification criteria should be determined by yourself after strictly reading the literature. It is not necessary to follow the scheme I specified. The classification should be based on the key influences on the reaction equation introduced in the literature
"""

    query = f"""
Could you please introduce what is the most crucial influencing factor in the base for suzuki chemical reactions
"""
    logger.info(f"Query is : {query}")
    run_query(rag_system, query)
