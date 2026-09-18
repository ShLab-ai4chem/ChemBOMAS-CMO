import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
import time
from torch.nn.utils import clip_grad_norm_
import json
import datetime
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from deepspeed import get_accelerator
import logging
import deepspeed
import subprocess
import wandb
from sklearn.metrics import r2_score
import numpy as np
import sys
import pathlib

import os
import re
import shutil

from . import cluster_utils

def init_deepspeed_distributed():
    # Initialize DeepSpeed
    deepspeed.init_distributed()#default:dist_backend='nccl'


# 仅单卡串行任务
# def get_device():
#     return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 多卡并行任务/仅单卡串行任务
def get_device():
    # 获取当前任务的 GPU 卡
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取当前任务的编号
    device_count = torch.cuda.device_count()  # 获取可用的 GPU 数量
    print(f'Local Rank:{local_rank}, Device Count:{device_count}')
    # 确保设备号在 GPU 数量范围内
    if device_count > 1:
        torch.cuda.set_device(local_rank)  # 设置当前任务使用的 GPU
        return torch.device(f'cuda:{local_rank}')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data_from_csv(path):
    data_df = pd.read_csv(path)
    dataset = Dataset.from_dict(data_df)
    return dataset

# lora_config = LoraConfig(
#     r=8,  # Rank of the low-rank matrix
#     lora_alpha=16,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj",],
#     #target_modules="all-linear",
#     # bias="none",
#     # modules_to_save=["classifier"]
#     )

def parse_output_string(output_val, num_targets=None):
    """
    解析逗号分隔的输出字符串为数值列表
    Args:
        output_val: 逗号分隔的字符串，如 "56,27" 或 "85.2"
        num_targets: 目标数量，如果为None则自动推断
    Return: 
        数值列表
    """
    # print(output_val)
    # print(type(output_val))
    # 处理可能的NaN或空值
    if pd.isna(output_val) or output_val == '' or output_val is None:
        return None
    
    # 确保是字符串类型
    if isinstance(output_val, torch.Tensor):
        output_val = str(float(output_val))
        
    if not isinstance(output_val, str):
        output_val = str(output_val)
    
    # 分割字符串并转换为浮点数
    try:
        values = [float(x.strip()) for x in output_val.split(',')]
    except ValueError as e:
        print(f"无法将输出值转换为浮点数: {output_val}")
        print(f"错误信息: {e}")
        return None
    
    # 如果指定了num_targets，检查长度
    if num_targets is not None and len(values) != num_targets:
        print(f"警告: 输出字符串应该有{num_targets}个值，但得到了{len(values)}个: {output_val}")
        return None
    
    # print(values)
    return values

def create_tensor_from_outputs(outputs, num_targets, device='cpu'):
    """
    从输出列表创建张量
    :param outputs: 包含多个样本输出的列表，每个样本是数值列表
    :param num_targets: 目标数量
    :param device: 设备
    :return: 形状为[batch_size, num_targets]的张量
    """
    # 过滤掉无效的输出
    valid_outputs = [out for out in outputs if out is not None]
    
    if not valid_outputs:
        raise ValueError("没有有效的输出数据")
    
    # 创建张量
    tensor = torch.tensor(valid_outputs, dtype=torch.float32, device=device)
    
    # 检查形状
    if num_targets == 1:
        # 单目标：确保形状正确
        if tensor.dim() == 2:
            tensor = tensor.squeeze(1)
    else:
        # 多目标：确保形状正确
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)
    
    return tensor

class YieldPredLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        产率预测层，支持单目标和多目标
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层维度
        :param output_size: 输出目标数量，默认为1（单目标）
        """
        super(YieldPredLayer, self).__init__()
        self.output_size = output_size
        self.act = nn.SiLU()
        self.predictor = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            # TODO：确定以下中间层是否需要
                            # self.act,
                            # nn.Linear(hidden_size, hidden_size // 2),
                            # self.act,
                            # nn.Linear(hidden_size // 2, output_size),
                            nn.Linear(hidden_size, output_size),
                        )
        
    def forward(self, x):
        pred = self.predictor(x)
        return pred
    
class LlamaWithLoss(nn.Module):
    def __init__(self, llama, predictor, num_targets=1):
        super(LlamaWithLoss, self).__init__()
        self.llama = llama
        self.num_targets = num_targets
        self.predictor = predictor
        
    def forward(self, inputs, y, pooling_method='last_token', return_loss=True):
        outputs = self.llama(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        
        if pooling_method == 'mean':
            embeddings = last_hidden_state.mean(dim=1)  # Mean pooling to get sentence-level embeddings
        elif pooling_method == 'last_token':
            embeddings = last_hidden_state[:,-1,:]
        else:
            raise ValueError("pooling_method must be 'mean' or 'last_token'")

        if return_loss:
            pred = self.predictor(embeddings)
            # 计算多目标损失：为每个目标计算独立的MSE损失然后求和
            if self.num_targets > 1:
                # y的形状应为 [batch_size, num_targets]
                # pred的形状为 [batch_size, num_targets]
                loss = torch.zeros(1, device=y.device)
                for i in range(self.num_targets):
                    loss += torch.nn.functional.mse_loss(pred[:, i], y[:, i])
            else:
                # 单目标情况
                loss = torch.nn.functional.mse_loss(pred.view(-1), y.view(-1))
            return embeddings, loss
        else:
            pred = self.predictor(embeddings)
            return embeddings, pred
    
    def eval(self):
        return super().eval()
    
    
def main(args):

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    
    logger.info(f'Base Model Path: {args.pretrained_model_path}')
    logger.info(f'SFT Model Path: {args.checkpoint_dir}')
    logger.info(f'Inference Result Path: {args.save_path}')
    logger.info(f'num of targets: {args.num_targets}')

    # 标识
    model_method = "pretrain" if "step1_llama3_8b_0916_yearly_pistachio_ep3" in args.pretrained_model_path else "no_pretrain"
    logger.info(f"Model Method: {model_method}")
    if "cluster" in args.checkpoint_dir:
        model_method = "cluster_" + model_method

    # Initialize DeepSpeed distributed environment
    init_deepspeed_distributed()
    device = get_device()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 获取当前任务的编号
    World_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1 
    logger.info(f"local_rank: {local_rank}; World_size: {World_size}")
    logger.info(f'[Rank {local_rank}] Starting inference on Device: {device}')
    
    output_state_dict_path = args.save_path
    os.makedirs(output_state_dict_path, exist_ok=True)
    
    # Construct the model and load the checkpoint
    if args.load_by_torch:
        base_model = cluster_utils.load_model_with_state_by_torch(args.pretrained_model_path).to(device)
        logger.info("Model load_by_torch")
    else:
        base_model = AutoModel.from_pretrained(args.pretrained_model_path).to(device)
        logger.info("Model load_by_pretrain")
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.lora:
        lora_config = LoraConfig(
            r=8,  # Rank of the low-rank matrix
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj",],
        )
        base_model = get_peft_model(base_model,lora_config)
    
    # 修改：使用args.num_targets初始化predictor
    predictor = YieldPredLayer(4096, 1024, args.num_targets).cuda()
    model = LlamaWithLoss(base_model, predictor, num_targets=args.num_targets)
    
    if args.checkpoint_dir:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)
        # 检查predictor的输出维度是否匹配
        if 'predictor.predictor.2.weight' in state_dict:
            loaded_output_size = state_dict['predictor.predictor.2.weight'].shape[0]
            if loaded_output_size != args.num_targets:
                print(f"警告: 加载的checkpoint输出维度({loaded_output_size})与指定目标数量({args.num_targets})不匹配")
                if args.num_targets == 1 and loaded_output_size > 1:
                    print("将使用第一个输出作为单目标预测")
                elif args.num_targets > 1 and loaded_output_size == 1:
                    print("将复制单目标输出作为多目标预测")
        model.load_state_dict(state_dict)
        logger.info("Model loaded")
    
    model.eval()
    
    # DeepSpeed inference configuration
    ds_inference_config = {
        "replace_with_kernel_inject": False,
        "tensor_parallel": {"tp_size": 1},
        "dtype": "fp32", #"fp16"
        "enable_cuda_graph": False
    }
    
    model_engine = deepspeed.init_inference(
        model=model,
        config=ds_inference_config
    )        
    
    logger.info("Distributed model is already on correct device")
    
    # Load the dataset
    data = read_data_from_csv(args.searchspace_path)
    
    # We can shard the data among ranks if we want each rank to handle different slices.
    data_sampler = DistributedSampler(
        data, 
        num_replicas=World_size, 
        rank=local_rank,
        shuffle=False    
    )
    dataloader = DataLoader(
        data, 
        batch_size=args.batch_size,  # Use the batch size from the arguments
        sampler=data_sampler,
        num_workers=8, 
        shuffle=False
    )
    logger.info("Dataset loaded")
    
    # Inference
    y_trues = []
    y_preds = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader,desc=f"Rank {local_rank} inference   ",file=sys.stdout)):    
            prompts = batch['instruction'] #获取输入指令
            
            # 处理输出数据
            output_data = batch['output']
            if isinstance(output_data, torch.Tensor):
                # 如果已经是张量，直接使用
                y = output_data.to(torch.float).to(device)
                
                # 确保形状正确
                if args.num_targets == 1:
                    # 单目标：确保是1D张量
                    if y.dim() > 1:
                        y = y.squeeze()
                else:
                    # 多目标：确保是2D张量
                    if y.dim() == 1:
                        y = y.unsqueeze(1)
                    
                    # 检查维度
                    if y.size(1) != args.num_targets:
                        raise ValueError(f"输出张量形状{y.shape}与目标数量{args.num_targets}不匹配")
            else:
                # 如果是其他格式（如字符串），使用解析逻辑
                outputs_list = []
                for output_val in output_data:
                    parsed_output = parse_output_string(output_val, args.num_targets)
                    outputs_list.append(parsed_output)
                
                y = create_tensor_from_outputs(outputs_list, args.num_targets, device)
            
            # 文本编码，对prompts进行编码
            inputs = tokenizer(prompts, max_length=3000, padding='longest', truncation=True, return_tensors="pt").to(device)
            # 将 inputs 字典中的每一个张量移动到当前进程对应的GPU上
            inputs = {k: v.to(device) for k,v in inputs.items()}
            y = y.to(device)
            
            emb, pred = model_engine(inputs, y, return_loss=False)
            
            # 保存预测结果和真实值
            y_preds.append(pred.detach().to(device))
            y_trues.append(y.detach().to(device))
            
    # Gather results from all ranks to rank 0
    if torch.distributed.is_initialized():
        # 所有rank都需要计算自己的预测结果
        if y_preds:
            all_preds = torch.cat(y_preds, dim=0)
            all_labels = torch.cat(y_trues, dim=0)
        else:
            # 如果没有数据，创建空张量
            if args.num_targets == 1:
                all_preds = torch.zeros(0, device=device)
                all_labels = torch.zeros(0, device=device)
            else:
                all_preds = torch.zeros(0, args.num_targets, device=device)
                all_labels = torch.zeros(0, args.num_targets, device=device)
        
        # 获取当前rank的张量大小和维度信息
        if args.num_targets == 1:
            # 单目标：确保是1D张量
            if all_preds.dim() > 1:
                all_preds = all_preds.squeeze(1)
            if all_labels.dim() > 1:
                all_labels = all_labels.squeeze(1)
            
            # 收集数据大小
            local_size = torch.tensor([all_preds.size(0)], device=device)
        else:
            # 多目标：确保是2D张量
            if all_preds.dim() == 1:
                all_preds = all_preds.unsqueeze(1)
            if all_labels.dim() == 1:
                all_labels = all_labels.unsqueeze(1)
            
            # 收集数据大小
            local_size = torch.tensor([all_preds.size(0), all_preds.size(1)], device=device)
        
        if local_rank == 0:
            # rank 0收集所有rank的张量大小
            if args.num_targets == 1:
                tensor_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(World_size)]
            else:
                tensor_sizes = [torch.zeros(2, dtype=torch.long, device=device) for _ in range(World_size)]
            
            dist.gather(local_size, gather_list=tensor_sizes, dst=0)
            
            # 创建gather_list
            max_seq_len = max([size[0].item() for size in tensor_sizes])
            
            if args.num_targets == 1:
                # 单目标：1D张量
                gathered_preds = [torch.zeros(max_seq_len, device=device) for _ in range(World_size)]
                gathered_labels = [torch.zeros(max_seq_len, device=device) for _ in range(World_size)]
            else:
                # 多目标：2D张量
                gathered_preds = [torch.zeros(max_seq_len, args.num_targets, device=device) for _ in range(World_size)]
                gathered_labels = [torch.zeros(max_seq_len, args.num_targets, device=device) for _ in range(World_size)]
            
            # 收集数据
            dist.gather(all_preds, gather_list=gathered_preds, dst=0)
            dist.gather(all_labels, gather_list=gathered_labels, dst=0)
            
            # 处理收集到的数据，移除填充部分
            valid_preds = []
            valid_labels = []
            
            for i, size in enumerate(tensor_sizes):
                seq_len = size[0].item()
                if seq_len > 0:
                    if args.num_targets == 1:
                        valid_preds.append(gathered_preds[i][:seq_len])
                        valid_labels.append(gathered_labels[i][:seq_len])
                    else:
                        valid_preds.append(gathered_preds[i][:seq_len])
                        valid_labels.append(gathered_labels[i][:seq_len])
            
            if valid_preds:
                y_preds_gathered = torch.cat(valid_preds, dim=0)
                y_labels_gathered = torch.cat(valid_labels, dim=0)
            else:
                y_preds_gathered = None
                y_labels_gathered = None

        else:
            # 其他rank发送数据大小和数据
            dist.gather(local_size, gather_list=None, dst=0)
            dist.gather(all_preds, gather_list=None, dst=0)
            dist.gather(all_labels, gather_list=None, dst=0)
                        
        # Only rank 0 will have the gathered results
        if local_rank == 0 and y_preds_gathered is not None:
            # Saving embeddings and predictions
            logger.info('Saving embeddings and predictions...')
            
            # 保存预测结果
            data_info_dict = {
                "pred_yields_by_rxn": y_preds_gathered.cpu(),
                "true_yields_by_rxn": y_labels_gathered.cpu() if y_labels_gathered is not None else None,
                "num_targets": args.num_targets
            }
            
            # 计算评估指标（如果真实值存在）
            if y_labels_gathered is not None:
                y_pred_np = y_preds_gathered.cpu().numpy()
                y_true_np = y_labels_gathered.cpu().numpy()
                
                if args.num_targets == 1:
                    # 单目标：计算MSE和R²
                    mse = np.mean((y_pred_np - y_true_np) ** 2)
                    r2 = r2_score(y_true_np, y_pred_np)
                    logger.info(f"单目标评估 - MSE: {mse:.4f}, R²: {r2:.4f}")
                    
                    # 保存评估指标
                    data_info_dict["metrics"] = {
                        "mse": float(mse),
                        "r2": float(r2)
                    }
                else:
                    # 多目标：为每个目标计算指标
                    metrics = {}
                    for i in range(args.num_targets):
                        mse = np.mean((y_pred_np[:, i] - y_true_np[:, i]) ** 2)
                        r2 = r2_score(y_true_np[:, i], y_pred_np[:, i])
                        
                        target_name = f"target_{i}"
                        if args.output_target_names and i < len(args.output_target_names):
                            target_name = args.output_target_names[i]
                        
                        metrics[target_name] = {
                            "mse": float(mse),
                            "r2": float(r2)
                        }
                        logger.info(f"目标 {target_name} - MSE: {mse:.4f}, R²: {r2:.4f}")
                    
                    # 计算平均指标
                    avg_mse = np.mean([m["mse"] for m in metrics.values()])
                    avg_r2 = np.mean([m["r2"] for m in metrics.values()])
                    metrics["average"] = {
                        "mse": float(avg_mse),
                        "r2": float(avg_r2)
                    }
                    logger.info(f"平均指标 - MSE: {avg_mse:.4f}, R²: {avg_r2:.4f}")
                    
                    data_info_dict["metrics"] = metrics
                    data_info_dict["output_target_names"] = args.output_target_names
            
            torch.save(data_info_dict, os.path.join(output_state_dict_path, f'{model_method}_yields.pt'))
            
            # 可选：保存为CSV格式便于查看
            try:
                if args.num_targets == 1:
                    df_results = pd.DataFrame({
                        'prediction': y_preds_gathered.cpu().numpy().flatten(),
                        'true_value': y_labels_gathered.cpu().numpy().flatten() if y_labels_gathered is not None else [None] * len(y_preds_gathered)
                    })
                else:
                    # 多目标：为每个目标创建列
                    pred_data = y_preds_gathered.cpu().numpy()
                    true_data = y_labels_gathered.cpu().numpy() if y_labels_gathered is not None else None
                    
                    df_dict = {}
                    for i in range(args.num_targets):
                        target_name = f"pred_target_{i}"
                        if args.output_target_names and i < len(args.output_target_names):
                            target_name = f"pred_{args.output_target_names[i]}"
                        df_dict[target_name] = pred_data[:, i]
                        
                        if true_data is not None:
                            true_name = f"true_target_{i}"
                            if args.output_target_names and i < len(args.output_target_names):
                                true_name = f"true_{args.output_target_names[i]}"
                            df_dict[true_name] = true_data[:, i]
                    
                    df_results = pd.DataFrame(df_dict)
                
                csv_path = os.path.join(output_state_dict_path, f'{model_method}_predictions.csv')
                df_results.to_csv(csv_path, index=False)
                logger.info(f"预测结果已保存为CSV: {csv_path}")
                
            except Exception as e:
                logger.error(f"保存CSV文件时出错: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='bert-base-uncased', help="The path to the pretrained model")
    parser.add_argument("--searchspace_path", type=str, default='tongji_searchspace_v2_97020', help="The name of the search space.")
    parser.add_argument("--lora", type=int, help="Use LoRA")
    parser.add_argument("--checkpoint_dir", type=str, help="The path to the checkpoint directory")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--load_by_torch", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    
    # ========= 多目标参数 =========
    parser.add_argument("--num_targets", type=int, default=1, help="Number of regression targets (default: 1 for single-target)")
    parser.add_argument("--output_target_names", type=str, nargs='+', default=None, help="Names of output targets (optional)")
    
    args = parser.parse_args()
    main(args)
