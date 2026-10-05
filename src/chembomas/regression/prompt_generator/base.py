from abc import ABC, abstractmethod

class BasePromptGenerator(ABC):
    """基础指令生成器抽象类"""
    
    @abstractmethod
    def generate_instruction(self, row_data: dict) -> str:
        """
        根据数据行生成指令
        
        Args:
            row_data: 包含所有列数据的字典
            
        Returns:
            生成的指令字符串
        """
        pass

