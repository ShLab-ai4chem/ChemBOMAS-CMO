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

import pathlib

import os
import re
import shutil, yaml

from . import cluster_utils

def parse_filename(filename):
    """
    解析文件名，提取 epoch, step, loss
    :param filename: 文件名字符串
    :return: (epoch, step, loss) 元组
    """
    match = re.match(r'(\d+)-(\d+)-(\d+\.?\d*)', filename)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        loss = float(match.group(3))
        return epoch, step, loss
    return None


def delete_files_wrt_loss(save_path, max_save_files=5, reverse=True):
    """
    保留 loss 最小的 k 个文件，并删除其他文件
    :param directory: 文件夹路径
    :param k: 保留的文件数量
    """
    files = os.listdir(save_path)
    parsed_files = []

    for file in files:
        parsed = parse_filename(file)
        if parsed:
            parsed_files.append((file, *parsed))

    if not parsed_files:
        return

    # 按 loss 排序
    parsed_files.sort(key=lambda x: x[3], reverse=reverse)
    print(parsed_files)
    # 保留前 k 个文件
    max_save_files = min(max_save_files, len(parsed_files))
    best_files = parsed_files[:max_save_files]

    # 删除其他文件
    for file, _, _, _ in parsed_files[max_save_files:]:
        file_path = os.path.join(save_path, file)
        for subfile in os.listdir(file_path):
            os.remove(os.path.join(file_path, subfile))
        os.rmdir(file_path)

    print(f"Kept {len(best_files)} best files with the smallest loss.")




def read_reaction(path,data_name):
    dataset = Dataset.load_from_disk(os.path.join(path, data_name))
    # raw_data = pd.read_csv(os.path.join(path, data_name, data_name + ".csv"))
    known_yields = dataset['yield']
    known_conditions = dataset['condition']
    reactions = dataset['reaction']
    return known_conditions, known_yields, reactions



def cleanup():
    dist.destroy_process_group()
    
def get_attr_string_from_json_files(json_files_dir):
    """
    convert attrs in json to attr_string
    :param json_files_dir: json files dir
    :return: attr_string
    """
    results = {}
    for file in os.listdir(json_files_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(json_files_dir, file)
        with open(file_path, "r") as f:
            json_data = json.load(f)
        data_maps = {}
        for data in json_data:
            name = data["name"]
            attrs = list(data.keys())
            att_string = ""
            for attr in attrs:
                if attr != "name":
                    att_string += f"{attr}: {data[attr]}\n"
            data_maps[name] = att_string
        
        results[file.split(".")[0]] = data_maps

    return results

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
    def __init__(self, llama, predictor, num_targets=1, loss_scale=None):
        super(LlamaWithLoss, self).__init__()
        self.llama = llama
        self.predictor = predictor
        self.num_targets = num_targets
        self.loss_scale = loss_scale if loss_scale is not None else [1.0] * num_targets
        
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
                    loss += torch.nn.functional.mse_loss(pred[:, i], y[:, i]) / self.loss_scale[i]
            else:
                # 单目标情况
                loss = torch.nn.functional.mse_loss(pred.view(-1), y.view(-1))
            return embeddings, loss
        else:
            pred = self.predictor(embeddings)
            return embeddings, pred

def read_data_from_csv(path):
    """
    从CSV文件读取数据，支持多目标输出
    """
    data_df = pd.read_csv(path)
    dataset = Dataset.from_dict(data_df)
    return dataset

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

def train(args, cluster_args=None):
    if args.local_rank == 0:
        if args.wandb_disable:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            os.environ["WANDB_DISABLED"] = "false"

            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            else:
                os.environ["WANDB_MODE"] = "online"

            if args.use_cluster:
                wandb.init( 
                    project=args.project_name,
                    name=f"{args.data_name}_cluster_{cluster_args['cluster_loss_lambda']}_B{args.per_device_train_batch_size}_E{args.num_epoch}_T{args.num_targets}",
                    )
            else:
                wandb.init( 
                    project=args.project_name,
                    name=f"{args.data_name}_B{args.per_device_train_batch_size}_E{args.num_epoch}_T{args.num_targets}",
                    )
    
    # 记录目标数量
    num_targets = args.num_targets
    if args.local_rank == 0:
        print(f"num of targets: {num_targets}")
        if args.output_target_names:
            print(f"names of targets: {args.output_target_names}")
    
    # Load the model and tokenizer
    pretrained_model_path = args.pretrained_model_path
    num_epoch = args.num_epoch
    batch_size = args.per_device_train_batch_size
    yield_predictor_path = args.yield_predictor_path
    lr = args.lr
    max_length = args.max_length

    data_path = args.data_path
    data_name = args.data_name
    # Save the base model
    lora_adapter_path = args.lora_adapter_path

    load_ds_dir = args.load_ds_dir
    load_ds_ckpt_id = args.load_ds_ckpt_id

    use_lora = args.use_lora

    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    
    logger.info(args.latest_method_notes)
    logger.info(f"num of targets: {num_targets}")
    if args.output_target_names:
        logger.info(f"names of targets: {args.output_target_names}")
    if args.use_cluster:
        logger.info(cluster_args)

    # args.global_rank = torch.distributed.get_rank()
    get_accelerator().set_device(args.local_rank)
    device = torch.device(get_accelerator().device_name(), args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    print('using device', device)


    if use_lora:

        # Define LoRA configuration
        if not os.path.exists(lora_adapter_path):
            if args.load_by_torch:
                print('Load model By Torch...')
                logger.info('Load model...')
                model = cluster_utils.load_model_with_state_by_torch(pretrained_model_path)
            else:
                print('Load model From Pretrain...')
                logger.info('Load model...')
                model = AutoModel.from_pretrained(pretrained_model_path, local_files_only=True, trust_remote_code=True)
                
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, local_files_only=True, trust_remote_code=True)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            # print(model)
            # logger.info(model)
            # Apply LoRA to the model\
            print('LoRA configuring...')
            logger.info('LoRA configuring...')
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=None,
                inference_mode=False,
                init_lora_weights=True,
                fan_in_fan_out=False,
                peft_type="LORA",
                revision=None,
                use_dora=False,
                use_rslora=False,
            )
            model = get_peft_model(model,lora_config)
        else:
            print(f"Load LoRA from {lora_adapter_path}")
            logger.info(f"Load LoRA from {lora_adapter_path}")
            lora_config = PeftConfig.from_pretrained(lora_adapter_path)
            if args.load_by_torch:
                model = cluster_utils.load_model_with_state_by_torch(lora_config.base_model_name_or_path)
            else:
                model = AutoModel.from_pretrained(lora_config.base_model_name_or_path, local_files_only=True)
                
            tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path, local_files_only=True)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        model.print_trainable_parameters()  # Print the number of trainable parameters to confirm LoRA is applied
    
    world_size = torch.distributed.get_world_size()
    rank = args.local_rank


    predictor = YieldPredLayer(4096, 1024, num_targets).to(device).train() 
    if os.path.exists(yield_predictor_path):
        try:
            # 加载predictor状态，注意处理输出维度不匹配的情况
            checkpoint = torch.load(yield_predictor_path, map_location=device)
            if num_targets != checkpoint.get('output_size', 1):
                print(f"警告: 加载的predictor输出维度({checkpoint.get('output_size', 1)})与当前设置({num_targets})不匹配")
                print("将重新初始化predictor的最后一层")
                # 只加载兼容的部分
                predictor.load_state_dict(checkpoint, strict=False)
            else:
                predictor.load_state_dict(checkpoint)
        except Exception as e:
            print("Error when load predictor : ", e)
            logger.info(f"Error when load predictor : {e}")
            print("将使用随机初始化的predictor")
    
    model = LlamaWithLoss(model, predictor, num_targets=num_targets, loss_scale=args.loss_scale)


    


    # Data
    print('Load data from...', os.path.join(data_path, data_name, 'train.csv'))
    logger.info(f'Load data ...')
    # train_data = Dataset.load_from_disk(os.path.join(data_path, data_name))
    train_data = read_data_from_csv(os.path.join(data_path, data_name, 'train.csv'))
    # DDP sampler
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    trainloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    
    # TODO: Cluster Data
    if cluster_args is not None:
        total_attr_strings_maps = get_attr_string_from_json_files(cluster_args['cluster_json_dir_path'])
        total_inputs_maps = {}
        for name, attr_strings_maps in total_attr_strings_maps.items():
            inputs_maps = {}
            for item, attr_string in attr_strings_maps.items():
                inputs = tokenizer(attr_string, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").to(device)
                inputs_maps[item] = inputs
            total_inputs_maps[name] = inputs_maps
        """
        total_inputs_maps格式: {'suzuki' : {...}, ...}
        inputs_maps格式: {'KOH' : ..., 'NaHCO3': ...}
        """

    # 
    optimizer_params = [
        {'params': model.llama.parameters(), 'lr': lr},
        {'params': model.predictor.parameters(), 'lr': lr*args.mlp_lr_multiplier}
    ]
    optimizer = optim.AdamW(optimizer_params)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    train_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    print("Train_batch_size: ", train_batch_size)
    logger.info(f'Train_batch_size: {train_batch_size}')

    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_batch_size'] = train_batch_size
    ds_config['scheduler']['params']['total_num_steps'] = num_epoch * len(trainloader) / args.gradient_accumulation_steps
    print(f"total effictive steps {ds_config['scheduler']['params']['total_num_steps']}")
    ds_config['scheduler']['params']['warmup_num_steps'] = ds_config['scheduler']['params']['total_num_steps']*0.1
    model, optimizer, _, _ = deepspeed.initialize(
        # args=args,
        model=model,
        optimizer=optimizer,
        config_params=ds_config,
        # model_parameters=all_parameters,
        # dist_init_required=True,
    )

    if load_ds_dir is not None and os.path.exists(load_ds_dir):
        print(f"Load deepspeed checkpoint from {load_ds_dir}")
        logger.info(f'Load deepspeed checkpoint from {load_ds_dir}')
        model.load_checkpoint(load_ds_dir, load_ds_ckpt_id)


    
    if args.run_test:
        test_data = read_data_from_csv(os.path.join(data_path, data_name, 'test.csv'))
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
        testloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, num_workers=8)

    best_eval_loss = torch.inf
    
    for epoch in range(1, num_epoch+1):
        print(f'Training Epoch {epoch}:')
        logger.info(f'Training Epoch {epoch}:')
        # DDP set epoch
        trainloader.sampler.set_epoch(epoch)

        total_loss = torch.scalar_tensor(0)
        losses = []
        run_time = 0
        for i, batch_data in enumerate(trainloader):
            model.train()
            model.llama.train()
            model.predictor.train()

            start_time = time.time()

            prompts = batch_data['instruction']
            
            # 解析输出字符串
            output_strings = batch_data['output']
            outputs_list = []
            
            for output_str in output_strings:
                parsed_output = parse_output_string(output_str, num_targets)
                outputs_list.append(parsed_output)
            
            # 创建目标张量
            y = create_tensor_from_outputs(outputs_list, num_targets, device)
            
            # add some noise
            noise = torch.randn_like(y).to(torch.float).to(device)
            y += noise
            
            inputs = tokenizer(prompts, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").to(device)

            # Get embeddings
            if model.fp16_enabled():
                y = y.half() 
            elif model.bfloat16_enabled():
                y = y.bfloat16()
            
            # TODO: add cluster
            _, pred_loss = model(inputs, y, pooling_method=args.pooling_method, return_loss=True)
            
            # TODO: Cluster
            distance_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            
            if args.use_cluster:
                # TODO: add cluster and new loss function
                distance_maps = {}
                for name, inputs_maps in total_inputs_maps.items():
                    # 遍历base, solvent, ligand
                    embed_maps = {}
                    for item, inputs in inputs_maps.items():
                        outputs = model.llama(**inputs, output_hidden_states=True)
                        last_hidden_state = outputs.last_hidden_state
                        if args.pooling_method == 'mean':
                            embeddings = last_hidden_state.mean(dim=1)  # Mean pooling to get sentence-level embeddings
                        elif args.pooling_method=='last_token':
                            embeddings = last_hidden_state[:,-1,:]
                        embed_maps[item] = embeddings   # data_maps
                    
                    if cluster_args['algorithm'] == "spectral":
                        cluster_cos_sim, extra_sims = cluster_utils.get_sim_matrix(embed_maps, use_extra=cluster_args['extra_name'], eps=torch.tensor(cluster_args['cluster_eps'], dtype=torch.float32))
                        distance = cluster_utils.cluster_similarity_virtual_point(
                            cluster_cos_sim, 
                            list(embed_maps.keys()), 
                            file_name=None, 
                            data_maps=embed_maps, 
                            extra_sims=extra_sims, 
                            extra_name=cluster_args['extra_name'],
                            use_angle_loss=cluster_args['use_angle_loss']
                        )
                    
                    else:
                        distance = cluster_utils.cluster_with_distance(
                            embed_maps, 
                            file_name=None, 
                            extra_name=cluster_args['extra_name'],
                            algorithm=cluster_args['algorithm']
                        )
                    
                    distance_maps[name] = distance
                
                # calculate distance_loss
                distance_loss = cluster_utils.calculate_distance_loss(distance_maps, alpha=cluster_args['cluster_weight_alpha'])
            
            # add distance_loss
            loss = cluster_args['cluster_loss_lambda'] * distance_loss + pred_loss if args.use_cluster else pred_loss
            losses.append(loss.cpu().item())
            model.backward(loss)
            model.step()
  

            end_time = time.time()
            run_time += (end_time-start_time)/ (60*60)
            print(f"Rank {rank}, Epoch {epoch}:{i+1}/{len(trainloader)}-step, \tloss:{loss.cpu().item():.2f}, \tpred_loss:{pred_loss.cpu().item():.2f}, \tdistance_loss:{distance_loss:.2f}, \truning time:{(end_time-start_time):.2f}s, \tleft_time:{((run_time/(i+1)) * (len(trainloader)-i-1) * 3600):.2f}s")
            logger.info(f"Rank {rank}, Epoch {epoch}:{i+1}/{len(trainloader)}-step, \tloss:{loss.cpu().item():.2f}, \tpred_loss:{pred_loss.cpu().item():.2f}, \tdistance_loss:{distance_loss:.2f}, \truning time:{(end_time-start_time):.2f}s, \tleft_time:{((run_time/(i+1)) * (len(trainloader)-i-1) * 3600):.2f}s ")

            if args.local_rank == 0 and (not args.wandb_disable): 
                wandb.log({
                    'train_loss': loss,
                    'distance_loss': distance_loss, 
                    'pred_loss': pred_loss, 
                    'current_lr': optimizer.param_groups[0]['lr']
                })

        

   
        if (epoch % args.save_interval == 0) or (epoch == num_epoch) or (epoch == 1):
            # Evaluate model
            model.eval()
            model.llama.eval()
            model.predictor.eval()
            eval_losses = []
            pred_all = []
            target_all = []
            best_r2_score = -1
            print(f"Evaluation ...")
            logger.info(f"Evaluation ...")
            
            if args.run_test:
                with torch.no_grad():
                    testloader.sampler.set_epoch(epoch)
                    for batch_data in tqdm(testloader):
                        prompts = batch_data['instruction']
                        
                        # 解析测试数据的输出字符串
                        output_strings = batch_data['output']
                        outputs_list = []
                        
                        for output_str in output_strings:
                            parsed_output = parse_output_string(output_str, num_targets)
                            outputs_list.append(parsed_output)
                        
                        # 创建目标张量
                        y_true = create_tensor_from_outputs(outputs_list, num_targets, device)

                        inputs = tokenizer(prompts, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").to(device)

                        # Get embeddings
                        if model.fp16_enabled():
                            y_true = y_true.half()
                        if model.bfloat16_enabled():
                            y_true = y_true.bfloat16()
                        
                        embeddings, pred = model(inputs, y_true, return_loss=False)
                        
                        # 保存预测结果和真实值用于计算评估指标
                        pred_all.append(pred.to(torch.float32).cpu().numpy())
                        target_all.append(y_true.to(torch.float32).cpu().numpy())
                        
                        # 计算评估损失
                        if args.loss_scale is None:
                            args.loss_scale = [1.0] * num_targets

                        if num_targets > 1:
                            # 多目标：为每个目标计算独立的MSE损失然后求和
                            eval_loss = torch.zeros(1, device=y_true.device)
                            for i in range(num_targets):
                                eval_loss += torch.nn.functional.mse_loss(pred[:, i], y_true[:, i]) / args.loss_scale[i]
                        else:
                            # 单目标
                            eval_loss = torch.nn.functional.mse_loss(pred.view(-1), y_true.view(-1)) / args.loss_scale[0]
                        
                        eval_losses.append(eval_loss.cpu().item())

            if len(eval_losses) == 0:
                if args.eval_save_ckpt:
                    ckpt_path = os.path.join(args.save_root, 'eval')
                    model.save_checkpoint(ckpt_path, tag=f'eval_{epoch}')
                    print(f"Model saved to: {ckpt_path}")
                    logger.info(f"Model saved to: {ckpt_path}")
                continue
            
            # 合并所有批次的预测结果和真实值
            pred_all = np.concatenate(pred_all, axis=0)
            target_all = np.concatenate(target_all, axis=0)
            
            # 计算评估指标
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            
            # 计算R²分数
            if num_targets == 1:
                # 单目标情况
                r2 = r2_score(target_all.astype(np.float64), pred_all.astype(np.float64))
                r2_scores = [r2]
                pred_all = pred_all.reshape(-1)
                target_all = target_all.reshape(-1)
            else:
                # 多目标情况：为每个目标计算R²分数
                r2_scores = []
                for i in range(num_targets):
                    r2 = r2_score(target_all[:, i].astype(np.float64), pred_all[:, i].astype(np.float64))
                    r2_scores.append(r2)
            
            # 计算平均R²分数
            avg_r2 = np.mean(r2_scores) if r2_scores else 0
            
            # 分布式环境下的指标聚合
            avg_eval_loss_tensor = torch.tensor(avg_eval_loss).to(torch.cuda.current_device())
            avg_r2_tensor = torch.tensor(avg_r2).to(torch.cuda.current_device())
            
            dist.all_reduce(avg_eval_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_r2_tensor, op=dist.ReduceOp.SUM)
            
            avg_eval_loss = avg_eval_loss_tensor.item() / world_size
            avg_r2 = avg_r2_tensor.item() / world_size
            
            print(f"Avg Eval Loss: {avg_eval_loss}, Avg R2: {avg_r2}")
            logger.info(f"Avg Eval Loss: {avg_eval_loss}, Avg R2: {avg_r2}")
            
            if args.local_rank == 0: 
                # 记录每个目标的R²分数
                if num_targets > 1:
                    print(f"各目标R²分数:")
                    logger.info(f"各目标R²分数:")
                    for i, r2_score_val in enumerate(r2_scores):
                        target_name = args.output_target_names[i] if args.output_target_names and i < len(args.output_target_names) else f"target_{i}"
                        print(f"  {target_name}: {r2_score_val}")
                        logger.info(f"  {target_name}: {r2_score_val}")
                        if not args.wandb_disable:
                            wandb.log({f'eval_R2_{target_name}': r2_score_val})
                
                if not args.wandb_disable:
                    wandb.log({'eval_loss': avg_eval_loss, 'eval_avg_R2': avg_r2})
                
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                
                if avg_r2 > best_r2_score:
                    best_r2_score = avg_r2
                    
                    if args.eval_save_ckpt:
                        ckpt_path = os.path.join(args.save_root, 'eval')
                        model.save_checkpoint(ckpt_path, tag=f'eval_{epoch}')
                        print(f"Model saved to: {ckpt_path}")
                        logger.info(f"Model saved to: {ckpt_path}")


        print(f"Avg Loss on Epoch {epoch}: {sum(losses) / len(losses)}")
        logger.info(f"Avg Loss on Epoch {epoch}: {sum(losses) / len(losses)}")
        

    ckpt_path = os.path.join(args.save_root, 'final')
    # model.save_checkpoint(ckpt_path, tag='final')
    # print(f"Model saved to: {ckpt_path}")

    os.makedirs(args.lora_adapter_save_path, exist_ok=True)
    model.save_checkpoint(args.lora_adapter_save_path, tag=args.data_name + args.latest_method_notes.split(' : ')[0])
    model.llama.save_pretrained(args.lora_adapter_save_path)
    
    # 保存predictor状态
    if args.save_predictor:
        predictor_state = predictor.state_dict()
        predictor_state['output_size'] = num_targets  # 保存输出维度信息
        torch.save(predictor_state, os.path.join(args.lora_adapter_save_path, "predictor.pt"))
        print(f"Predictor saved to: {args.lora_adapter_save_path}/predictor.pt")
    
    with open(os.path.join(args.lora_adapter_save_path, "notes.txt"), "w") as f:
        f.write(args.latest_method_notes)
        f.write(f"\nnum of targets: {num_targets}")
        if args.output_target_names:
            f.write(f"\nnames of targets: {', '.join(args.output_target_names)}")
    
    print(f"LoRA saved to: {args.lora_adapter_save_path}")


    # TODO: Still have bug    
    # if dist.get_rank() == 0:                # 只在 rank0 做合并/保存，避免重复
    #     with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
    #         peft_model = model.module
    #         merged_model = peft_model.merge_and_unload()
    #         torch.save(merged_model.predictor.state_dict(),
    #                 os.path.join(args.lora_adapter_save_path, "predictor.pt"))
    #         logger.info(f"Predictor saved to : {args.lora_adapter_save_path}")

    # logger.info(f"Model saved to: {ckpt_path}")
    if args.local_rank == 0 and (not args.wandb_disable): 
        wandb.log({'train_loss_on_epoch': total_loss.item() / (i+1)})
        wandb.finish()

        

def main():
    parser = argparse.ArgumentParser(description="Distributed Data Parallel Training for Multi-target Regression")
    
    parser.add_argument("--pretrained_model_path", required=True, help="local Hugging Face base-model path")
    parser.add_argument("--lora_adapter_path", default=None, help="optional local LoRA adapter path")
    parser.add_argument("--yield_predictor_path", default=None, help="optional local regression-head path")
    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_path", required=True, help="directory containing generated SFT CSV data")
    parser.add_argument("--data_name", default='FD_wqp')
    parser.add_argument("--save_root", default="outputs", help="directory for newly trained artifacts")
    parser.add_argument("--base_model_save_path", default="base_model")
    parser.add_argument("--lora_adapter_save_path", default="lora_adapter")
    parser.add_argument('--use_lora', type=int, default=1)
    parser.add_argument('--log_file', type=str, default="training_ds.log")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=3000)
    parser.add_argument('--load_ds_dir', type=str, default=None)
    parser.add_argument('--load_ds_ckpt_id', type=str, default=None)
    parser.add_argument('--mlp_lr_multiplier',type=float, default=10)
    parser.add_argument('--project_name', type=str, default="llama_regression")

    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--save_interval", type=int, default=20)
    # ========= New Add =========
    parser.add_argument("--eval_save_ckpt", type=int, default=0)
    parser.add_argument("--save_lora_adapter", type=int, default=1)
    parser.add_argument("--save_predictor", type=int, default=1)
    parser.add_argument("--pooling_method", type=str, default="last_token", choices=["mean", "last_token"])
    parser.add_argument("--run_test", type=int, default=1)
    parser.add_argument("--wandb_disable", type=int, default=1)
    parser.add_argument("--wandb_offline", type=int, default=0)
    parser.add_argument("--load_by_torch", type=int, default=0)
    parser.add_argument("--latest_method_notes", type=str, default="None method notes")
    
    # cluster params in yaml
    parser.add_argument("--use_cluster", type=int, default=1)
    parser.add_argument("--cluster_config", type=str, default="cluster_config/cluster_config.yaml")
    
    # ========= Multi-target Parameters =========
    parser.add_argument("--num_targets", type=int, default=1, help="Number of regression targets (default: 1 for single-target)")
    parser.add_argument("--output_target_names", type=str, nargs='+', default=None, help="Names of output targets (optional)")
    parser.add_argument("--loss_scale", type=float, nargs='+', default=None, help="Loss scaling factors for each target (optional, default uses 1.0 for all targets)")

    args = parser.parse_args()
    
    if args.use_cluster:
        cluster_args = yaml.load(open(args.cluster_config, 'r'), Loader=yaml.FullLoader)
    else:
        cluster_args = None

    os.makedirs(args.save_root, exist_ok=True)
    
    # world_size = args.world_size
    print('start training')
    print(f"Process local rank = {args.local_rank}")


    # deepspeed.launcher.executable.main(train, args=(world_size, args), nprocs=world_size)
    # mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    # os.environ['TOKENIZERS_PARALLELISM='] = "true"
    # os.environ['RANK'] = os.environ['SLURM_PROCID']
    # os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    # os.environ['MASTER_PORT'] = str(random.randint(1024, 65535))
    # os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
    train(args, cluster_args)


if __name__ == '__main__':

    main()
