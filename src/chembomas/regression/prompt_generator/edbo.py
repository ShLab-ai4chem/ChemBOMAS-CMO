from .base import BasePromptGenerator

class EDBOPromptGenerator(BasePromptGenerator):
    """EDBO数据集SFT指令生成器"""
    
    def generate_instruction(self, row_data: dict) -> str:
        """
        生成EDBO数据集的微调指令
        """
        # 提取字段
        ligand = row_data.get('Ligand_SMILES', '')
        solvent = row_data.get('Solvent_SMILES', '')
        base = row_data.get('Base', '')
        temp = row_data.get('Temperature', '')
        conc = row_data.get('Concentration', '')
        
        # 构建指令
        instruction = (
            f"Here is a chemical reaction. Reactants are: Cn1cnc(C#N)c1, Fc1ccccc1Br. "
            f"Product is: Cn1cnc(C#N)c1-c1ccccc1F. "
            f"Reaction type is Pd catalyzed C-H arylation. "
            "The reaction conditions of this reaction are: "
            f"Palladium Source: [PdCl(allyl)]2. "
            f"Ligand: {ligand}. "
            f"Solvent: {solvent}. "
            f"Base: {base}. "
            f"Reaction Temperature: {temp} degrees centigrade. "
            f'Reaction Concentration: {conc} M. '
            f"Palladium loading: 2.25 mol%. "
            f"Ligand loading: 5 mol%. "
            f"Base loading: 3 eq. "
            "What is the yield (%) and cost (US dollar per mol) of this reaction?"
        )
        return instruction