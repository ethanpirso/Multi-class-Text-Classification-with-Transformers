import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class NewsGroupsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['data']
        label = self.data.iloc[idx]['target']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(batch_size, max_len):
    data = fetch_20newsgroups(subset='train')
    df = pd.DataFrame({'data': data.data, 'target': data.target})
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])
    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

    train_dataset = NewsGroupsDataset(train_df, tokenizer, max_len)
    val_dataset = NewsGroupsDataset(val_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    test_data = fetch_20newsgroups(subset='test')
    test_df = pd.DataFrame({'data': test_data.data, 'target': test_data.target})
    test_dataset = NewsGroupsDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    num_classes = len(df['target'].unique())
    return train_loader, val_loader, test_loader, num_classes
