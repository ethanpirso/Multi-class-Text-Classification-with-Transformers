import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
import lightning as L
from torchmetrics.functional import accuracy, precision, recall, f1_score
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        return self.out(out)

class TransformerClassifier(L.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5):
        super(TransformerClassifier, self).__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        self.attention = MultiHeadAttention(embed_dim=self.bert.config.hidden_size, num_heads=2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        attention_output = self.attention(bert_output.last_hidden_state)
        pooled_output = torch.mean(attention_output, dim=1)
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.hparams.num_classes)
        precision_val = precision(preds, labels, task='multiclass', num_classes=self.hparams.num_classes, average='macro')
        recall_val = recall(preds, labels, task='multiclass', num_classes=self.hparams.num_classes, average='macro')
        f1_val = f1_score(preds, labels, task='multiclass', num_classes=self.hparams.num_classes, average='macro')     
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        self.log('test_precision', precision_val, prog_bar=True, logger=True)
        self.log('test_recall', recall_val, prog_bar=True, logger=True)
        self.log('test_f1', f1_val, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc, 'test_precision': precision_val, 'test_recall': recall_val, 'test_f1': f1_val}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
