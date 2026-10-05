from .fd_wqp import FDWQPPromptGenerator
from .edbo import EDBOPromptGenerator

# 可用的生成器映射
GENERATOR_REGISTRY = {
    'FD_wqp': FDWQPPromptGenerator,
    'EDBO': EDBOPromptGenerator,
}

def get_generator(generator_type: str, **kwargs):
    """
    获取指定类型的指令生成器
    
    Args:
        generator_type: 生成器类型
        **kwargs: 传递给生成器的参数
        
    Returns:
        指令生成器实例
        
    Raises:
        ValueError: 如果生成器类型不存在
    """
    if generator_type not in GENERATOR_REGISTRY:
        available = list(GENERATOR_REGISTRY.keys())
        raise ValueError(f"未知的生成器类型: {generator_type}。可用类型: {available}")
    
    return GENERATOR_REGISTRY[generator_type](**kwargs)

def list_available_generators():
    """列出所有可用的生成器类型"""
    return list(GENERATOR_REGISTRY.keys())

# 导出
__all__ = [
    'FDWQPPromptGenerator',
    'EDBOPromptGenerator',
    'get_generator',
    'list_available_generators',
]
