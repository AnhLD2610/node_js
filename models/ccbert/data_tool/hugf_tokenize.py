'''
Author WHU ZFJ 2021
data pre- and post- processing
'''
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
from config import args
from data_tool.scorer import convention_tokenize
import csv

tokenizer = RobertaTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, example_id, des_ids, code_ids,
                 target_ids):
        self.example_id = example_id
        self.des_ids = des_ids
        self.code_ids = code_ids
        self.target_ids = target_ids

def read_features(filename, args):
    """
    Convert source data to features
    """
    features = []
    with open(filename, 'r',encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for idx, row in enumerate(csv_reader):
            body = row['body'].strip()
            code = row['code'].strip()
            title = row['title'].strip()
            example_id = idx
            
            des_tokens = tokenizer.tokenize(body)[:args.max_src_len-2]
            des_tokens = [tokenizer.cls_token] + des_tokens + [tokenizer.sep_token]
            des_ids = tokenizer.convert_tokens_to_ids(des_tokens)
            
            code_tokens = tokenizer.tokenize(code)[:args.max_src_len-2]
            code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
            
            target_tokens = tokenizer.tokenize(title)[:args.max_tgt_len-2]
            target_tokens = [tokenizer.bos_token] + target_tokens + [tokenizer.eos_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            features.append(InputFeatures(
                example_id, des_ids, code_ids, target_ids
            ))
    return features

def gen_batch(data_batch):
    '''
    Padding according to the longest sequence in the batch
    The padding is not long enough, and the truncation is too long
    Return the id sequence after padding 
    '''
    def _trunc_and_pad(seq_ids, max_len):
        # seq_ids SOS, EOS, UNK already included 
        eos_id = seq_ids[-1]
        seq_ids = seq_ids[:-1] # Delete EOS first 
        seq_ids = list(seq_ids[:max_len-1]) # Truncate data that is too long 
        seq_ids.append(eos_id) # Add EOS to the truncated data 
        seq_ids = torch.LongTensor(seq_ids)
        return seq_ids
        
    des_ids_for_padding = []
    code_ids_for_padding = []
    tgt_ids_for_padding = []
    
    for raw_des_ids_batch, raw_code_ids_batch, raw_tgt_ids_batch in data_batch:
        # raw_src_ids_batch [seq_len]
        des_ids_batch = _trunc_and_pad(raw_des_ids_batch, args.max_src_len)
        code_ids_batch = _trunc_and_pad(raw_code_ids_batch, args.max_src_len)
        target_ids_batch = _trunc_and_pad(raw_tgt_ids_batch, args.max_tgt_len)
        
        des_ids_for_padding.append(des_ids_batch)
        code_ids_for_padding.append(code_ids_batch)
        tgt_ids_for_padding.append(target_ids_batch)
        
    padded_des_ids = pad_sequence(des_ids_for_padding, padding_value=tokenizer.pad_token_id) # [seq_len, batch_size]
    padded_code_ids = pad_sequence(code_ids_for_padding, padding_value=tokenizer.pad_token_id) # [seq_len, batch_size]
    padded_tgt_ids = pad_sequence(tgt_ids_for_padding, padding_value=tokenizer.pad_token_id)
    
    # result = [i for i in zip(padded_src_ids, padded_tgt_ids)]
    des_padding_mask = torch.ones(padded_des_ids.shape) # The pad is 0 
    des_padding_mask[padded_des_ids==tokenizer.pad_token_id] = 0
    code_padding_mask = torch.ones(padded_code_ids.shape) # The pad is 0 
    code_padding_mask[padded_code_ids==tokenizer.pad_token_id] = 0
    tgt_padding_mask = torch.ones(padded_tgt_ids.shape) # The pad is 0 
    tgt_padding_mask[padded_tgt_ids==tokenizer.pad_token_id] = 0
    return padded_des_ids, des_padding_mask, padded_code_ids, code_padding_mask, padded_tgt_ids, tgt_padding_mask

def get_dataloaders(args):
    '''
    The ultimate encapsulation of the data layer
    read the divided data set, and convert it to the dataloader of torch 
    '''
    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    test_file_path = args.test_file_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    
    datasets = [] # train, val, test
    for data_file in [train_file_path, val_file_path, test_file_path]:
        features = read_features(data_file, args)
        des_ids = [torch.tensor(f.des_ids) for f in features]
        code_ids = [torch.tensor(f.code_ids) for f in features]
        target_ids = [torch.tensor(f.target_ids) for f in features]
        dataset = [i for i in zip(des_ids, code_ids, target_ids)]
        datasets.append(dataset)
    train_sampler = RandomSampler(datasets[0])
    train_dataloader = DataLoader(datasets[0], sampler=train_sampler, batch_size=train_batch_size, collate_fn=gen_batch)
    val_sampler = SequentialSampler(datasets[1])
    val_dataloader = DataLoader(datasets[1], sampler=val_sampler, batch_size=test_batch_size, collate_fn=gen_batch)
    test_sampler = SequentialSampler(datasets[2])
    test_dataloader = DataLoader(datasets[2], sampler=test_sampler, batch_size=test_batch_size, collate_fn=gen_batch)
    return train_dataloader, val_dataloader, test_dataloader


def decode_batch_ids(batch_seq_ids):
    '''
    batch_seq_ids [batch_size, seq_len]
    '''
    pred_tokens = []
    for seq in batch_seq_ids:
        token_ids = []
        for tok_id in seq:
            if tok_id == tokenizer.eos_token_id:
                break
            if tok_id == tokenizer.bos_token_id:
                break
            token_ids.append(tok_id.item())
        decode_tokens = tokenizer.decode(token_ids)
        decode_tokens = convention_tokenize(decode_tokens)
        pred_tokens.append(decode_tokens)
    return pred_tokens
