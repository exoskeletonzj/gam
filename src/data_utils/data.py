import os 
import json 

import torch 
from torch.utils.data import DataLoader, Dataset
import random

def my_collate(batch):
    '''
    'doc_key': ex['doc_key'],
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'tgt_token_ids': tgt_tokens['input_ids'],
    'tgt_attn_mask': tgt_tokens['attention_mask'],
    '''
    
    
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])
    input_metion = torch.stack([torch.LongTensor(ex['input_metion']) for ex in batch])

    input_entity = [ex['input_entity'] for ex in batch]
    # output_entdict = [ex['output_entdict'] for ex in batch]
    sentence_dict = [ex['sentence_dict'] for ex in batch]
    coref_dict = [ex['coref_dict'] for ex in batch]
    result = {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
        'input_metion':input_metion,
        'input_entity': input_entity,
        # 'output_entdict': output_entdict,
        'sentence_dict': sentence_dict,
        'coref_dict': coref_dict,
    }
    if 'event_idx' in batch[0]:
        event_idx = [ex['event_idx'] for ex in batch]
        result['event_idx'] = event_idx
    if 'event_type' in batch[0]:
        event_type = [ex['event_type'] for ex in batch]
        result['event_type'] = event_type
    if 'trigger' in batch[0]:
        trigger = [ex['trigger'] for ex in batch]
        result['trigger'] = trigger

    return result


def my_collate_training(batch):
    # 'event_idx': i, 
    # 'doc_key': ex['doc_id'], 
    # 'input_token_ids':input_tokens['input_ids'],
    # 'input_attn_mask': input_tokens['attention_mask'],
    # 'tgt_token_ids': tgt_tokens['input_ids'],
    # 'tgt_attn_mask': tgt_tokens['attention_mask'],
    # 'compare_token_ids': compare_tokens['input_ids'],
    # 'compare_attn_mask': compare_tokens['attention_mask'],
    # 'input_mask' : input_mask,
    # 'compare_mask':compare_mask

    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])


    input_mask = torch.stack([torch.BoolTensor(ex['input_mask']) for ex in batch])

    input_metion = torch.stack([torch.LongTensor(ex['input_metion']) for ex in batch])
    input_entity = [ex['input_entity'] for ex in batch]

    # output_entdict = [ex['output_entdict'] for ex in batch]
    sentence_dict=[ex['sentence_dict'] for ex in batch]
    coref_dict=[ex['coref_dict'] for ex in batch]
    # arg_metion = [ex['arg_metion'] for ex in batch]
    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'input_mask' : input_mask,
        'input_metion':input_metion,
        'input_entity': input_entity,
        'doc_key': doc_keys,
        # 'output_entdict': output_entdict,
        'sentence_dict': sentence_dict,
        'coref_dict':coref_dict,
        # 'arg_metion': arg_metion,
    }


class IEDataset(Dataset):
    def __init__(self, input_file, tokenizer = None):
        super().__init__()
        self.examples = []
        with open(input_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

