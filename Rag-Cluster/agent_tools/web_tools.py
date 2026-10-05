from langchain_community.utilities import GoogleSerperAPIWrapper
from loguru import logger
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os

search = GoogleSerperAPIWrapper() if os.environ.get("SERPER_API_KEY") else None

# 定义符合Agent要求的工具格式
class SearchInput(BaseModel):
    search_query: str = Field(..., description="The search query for Google search")

def google_search(search_query: str) -> str:
    """执行Google搜索并返回结果"""
    if search is None:
        raise RuntimeError("Set SERPER_API_KEY before enabling web search")
    return search.run(search_query)

# 创建结构化工具实例
google_search_tool = StructuredTool.from_function(
    func=google_search,
    name="GoogleSearch",
    description="Use Google search to get the latest data",
    args_schema=SearchInput,
    return_direct=True  # 直接返回原始结果
)

if __name__ == "__main__":
    # 测试工具
    query = "NaHCO3 pH"
    result = google_search_tool.invoke({"search_query": query})
    logger.info(f"Google搜索结果: {result}")
