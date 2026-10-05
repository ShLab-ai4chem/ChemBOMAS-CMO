from .base import BasePromptGenerator

class FDWQPPromptGenerator(BasePromptGenerator):
    """FD_wqp项目SFT指令生成器"""
    
    def generate_instruction(self, row_data: dict) -> str:
        """
        生成FD_wqp项目的指令
        """
        # 提取字段
        Pd_source = row_data.get('Palladium_Source', '')
        ligand = row_data.get('Chiral_Ligand_SMILES', '')
        solvent = row_data.get('Solvent', '')
        additive = row_data.get('Additive', '')
        base = row_data.get('Base', '')
        temp = row_data.get('Temperature', '')
        time = row_data.get('Time', '24h')
        ligand_load = row_data.get('Chiral_Ligand_Loading', '20')  # mol%
        if additive == 'Nothing':
            additive_load = 'Nothing'
        else:
            additive_load = f"{row_data.get('Additive_Loading', '1.0')} eq"
        base_load = row_data.get('Base_Loading', '3.0')  # eq.
        
        # 构建指令
        instruction = (
            f"Here is a chemical reaction. Reactants are: C[Si](C)(C)C(Cl)c1ccccc1, COc1ccc(B(O)O)cc1. "
            f"Product is: COc1ccc(C(c2ccccc2)[Si](C)(C)C)cc1. "
            f"Reaction type is chiral Suzuki Coupling. "
            "The reaction conditions of this reaction are: "
            f"Palladium Source: {Pd_source}. "
            f"Chiral Ligand: {ligand}. "
            f"Solvent: {solvent}. "
            f"Additive: {additive}. "
            f"Base: {base}. "
            f"Reaction Temperature: {temp} degrees centigrade. "
            f"Reaction Time: {time}. "
            f"Substrate ratio: 3:1 . "
            f"Palladium loading: 10 mol%. "
            f"Ligand loading: {ligand_load} mol%. "
            f"Additive loading: {additive_load}. "
            f"Base loading: {base_load} eq. "
            f"Solvent volume: 1 mL. "
            "What is the yield and enantioselectivity(ee%) of this reaction?"
        )
        return instruction