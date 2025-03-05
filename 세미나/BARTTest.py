from transformers import BartModel, BartTokenizer, BartConfig


model = BartModel.from_pretrained('facebook/bart-large')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


# configuration = BartConfig()
print(model)
