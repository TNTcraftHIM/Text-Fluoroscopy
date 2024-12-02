from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import pandas as pd

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device='cpu'), sequence_lengths]

max_length = 512

pretrained_model_name_or_path = 'huggingface_model/gte-Qwen1.5-7B-instruct'
which_embedding='gte-qwen_all_embedding'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,device_map='auto')
save_dir = f'save/{which_embedding}/save_embedding/'
labels_dir = f'dataset/labels/'

def get_all_embedding(model,input_texts):
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**batch_dict,output_hidden_states=True)
    all_embed = [last_token_pool(outputs.hidden_states[i].cpu(), batch_dict['attention_mask']) for i in range(len(outputs.hidden_states))]
    all_embed = torch.concat(all_embed,1).cpu()
    return all_embed

data_dir = 'data/'
test_datasets = {}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        print(file_name)
        result_data = []
        test_datasets[file_name] = {'data':[],'label':[]}
        data = pd.read_csv(data_dir+file_name, encoding='utf-8')
        embeddings = []
        results = []
        for _, text_info in tqdm(data.iterrows(), total=data.shape[0]):
            text = text_info['text']
            result = 1 if 'MGT' in text_info['label'] else 0
            prompt = text
            embedding = get_all_embedding(model,[text])
            embeddings.append(embedding)
            results.append(result)
            if len(embeddings) >=300:
                break
        embeddings = torch.cat(embeddings, dim=0)
        results = torch.tensor(results)
        print('embedding shape:',embeddings.shape)
        print(save_dir+file_name.split('.')[0]+'.pt')
        torch.save(embeddings, save_dir+file_name.split('.')[0]+'.pt')
        torch.save(results, labels_dir+file_name.split('.')[0]+'.pt')
