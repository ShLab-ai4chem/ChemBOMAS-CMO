from langchain.tools import Tool
from loguru import logger
import requests
import urllib.parse
from typing import Dict, List, Union
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}

def find_property_in_json(data: Union[Dict, List], property_name: str):
    """
    递归地在PubChem返回的复杂JSON结构中搜索特定属性。

    Args:
        data: 要搜索的JSON数据部分 (字典或列表)。
        property_name: 要查找的属性标题，例如 'pH', 'Melting Point'。

    Returns:
        找到的属性信息字符串，或None。
    """
    if isinstance(data, dict):
        # 检查当前字典是否是我们寻找的目标部分
        # 典型的结构是 {'TOCHeading': 'pH', 'Information': [...]}
        if data.get('TOCHeading') and data.get('TOCHeading').lower() == property_name.lower():
            # 尝试提取信息，PubChem的数据格式不统一，需要尝试多种可能
            if 'Information' in data and data['Information']:
                info = data['Information'][0]
                if 'Value' in info and 'StringWithMarkup' in info['Value']:
                    return info['Value']['StringWithMarkup'][0].get('String')
                elif 'Value' in info and 'Number' in info['Value']:
                     return str(info['Value']['Number'][0])

            # 另一种可能的结构
            if 'Text' in data:
                return ' '.join(data['Text'])

        # 如果不是目标，则继续递归搜索其值
        for key, value in data.items():
            result = find_property_in_json(value, property_name)
            if result:
                return result

    elif isinstance(data, list):
        # 如果是列表，则递归搜索其中每个元素
        for item in data:
            result = find_property_in_json(item, property_name)
            if result:
                return result

    return None

def get_chem_info(substance_name: str, property_name: str):
    normalized_name = substance_name.lower().strip()
    encoded_name = urllib.parse.quote(normalized_name)

    logger.info(f"正在查询物质 '{substance_name}' 的CID...")

    try:
        # 步骤1: 通过PubChem API获取CID (化合物ID)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        cid_response = requests.get(cid_url, timeout=10)

        if cid_response.status_code != 200:
            logger.warning(f"获取CID失败，状态码: {cid_response.status_code} for {substance_name}")
            return f"Cannot find CID for substance '{substance_name}'."

        cid_data = cid_response.json()
        cid_list = cid_data.get('IdentifierList', {}).get('CID', [])
        if not cid_list:
            logger.warning(f"在PubChem中未找到物质 '{substance_name}'。")
            return f"Substance '{substance_name}' not found in PubChem."

        cid = cid_list[0]
        logger.info(f"找到CID: {cid}。正在获取详细属性...")

        # 步骤2: 根据CID获取完整的物理化学属性
        physchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        physchem_response = requests.get(physchem_url, timeout=15)

        if physchem_response.status_code != 200:
            logger.error(f"获取属性失败，状态码: {physchem_response.status_code}")
            return "Failed to retrieve properties, server returned an error."

        physchem_data = physchem_response.json()

        # 步骤3: 使用递归函数在返回的JSON中搜索所需属性
        logger.info(f"正在从返回的数据中搜索属性 '{property_name}'...")
        result = find_property_in_json(physchem_data, property_name)

        if result:
            logger.success(f"成功找到属性 '{property_name}': {result}")
            return f"{substance_name} {property_name}: {result}"
        else:
            logger.warning(f"在 '{substance_name}' 的信息中未找到属性 '{property_name}'。")
            return f"Property '{property_name}' not found for '{substance_name}'."

    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return "Network request failed"
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"解析JSON数据时出错: {e}")
        return "Error parsing response data"

class ChemInfoInput(BaseModel):
    substance_name: str = Field(
        ...,
        description="The name or chemical formula of the substance. For example: 'water', 'H2SO4', 'KOH'."
    )
    property_name: str = Field(
        ...,
        description="The specific physical or chemical property to find. For example: 'pH', 'boiling point', 'molecular weight'."
    )

chem_database_search = StructuredTool.from_function(
    func=get_chem_info,
    name="chem_database_search",
    description="Use this tool to find a specific physical property of a chemical substance. You must extract both the substance's name and the property's name from the user's query.",
    args_schema=ChemInfoInput
)

if __name__ == '__main__':
    # 示例1: 查询KOH的pH值
    print("\n--- 查询KOH的pH ---")
    output = chem_database_search.invoke({'substance_name': 'KOH', 'property_name': 'pH'})
    print(output)