from collections import OrderedDict
import torch
from transformers import AutoConfig, AutoModel

def load_state_by_torch(weights_path):
    state_dict = torch.load(weights_path + '/pytorch_model.bin')

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[6:]] = v

    return new_sd

def load_model_with_state_by_torch(weights_path):
    config = AutoConfig.from_pretrained(weights_path, trust_remote_code=True)
    model = AutoModel.from_config(config)
    state_dict = torch.load(weights_path + '/pytorch_model.bin')

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[6:]] = v

    model.load_state_dict(new_sd)
    return model