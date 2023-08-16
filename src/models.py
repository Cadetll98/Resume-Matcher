# Libraries Required:
# -- spacy (pip install spacy && python -m spacy download en_core_web_md)
# -- numpy
# -- fasttext
# -- tensorflow
# -- tensorflow_hub
# -- transformers
# -- torch

# Linux or MacOS required
import spacy
import numpy as np
import fasttext
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from transformers import GPT2Tokenizer, GPT2Model

INPUT_FOLDER = '../data'

''' GloVe '''
glove_model = spacy.load('en_core_web_md')

def word_embedding_glove(sentence):
  return glove_model(sentence).vector


''' FastText '''
fasttext_model = fasttext.load_model(INPUT_FOLDER+'/cc.en.300.bin')

def word_embedding_fasttext(sentence):
    words = sentence.lower().split()
    embeddings = [fasttext_model.get_word_vector(word) for word in words]
    return np.mean(embeddings, axis=0)
  

''' ELMo '''
elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
tf.compat.v1.enable_eager_execution()

def word_embedding_elmo(sentence):
    embeddings = elmo_model.signatures["default"](tf.constant([sentence]))["default"]
    return embeddings.numpy().squeeze()
  

''' BERT '''
model_name = 'bert-large-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def word_embedding_bert(sentence):
  tokens = bert_tokenizer.encode(sentence, add_special_tokens=True, max_length=1024, truncation=True)
  ids = torch.tensor(tokens).unsqueeze(0)
  with torch.no_grad():
    return bert_model(ids).last_hidden_state.mean(dim=1).squeeze()


''' GPT '''
model_name = 'gpt2'
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt_model = GPT2Model.from_pretrained(model_name)
gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def word_embedding_gpt(sentence):
    encoded = gpt_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
      outputs = gpt_model(**encoded)
      return outputs.last_hidden_state.mean(dim=1).numpy()
