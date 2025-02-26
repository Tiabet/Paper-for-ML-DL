from transformers import BertTokenizer, BertForPreTraining, BertConfig
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

configuration = BertConfig()
print(configuration)