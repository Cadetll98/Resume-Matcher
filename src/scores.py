# Libraries Required:
# -- sklearn
# -- numpy
# -- scipy

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.spatial.distance import cityblock

models = ['word2vec', 'fasttext', 'glove', 'elmo', 'bert', 'gpt']

''' Cosine Similarity '''
# [-1, 1]. higher value -> higher similarity
def get_cosine_similarity(job_embedding, resume_embedding, model):
  
  if model not in models:
    raise Exception('Model Not Found!')

  if model in ['fasttext', 'glove', 'bert']:
    return cosine_similarity(job_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0, 0]
    
  elif model == 'elmo':
    return cosine_similarity([job_embedding], [resume_embedding])[0][0]

  # GPT model
  else:
    return cosine_similarity(job_embedding, resume_embedding)[0][0]
  
  
''' Eucledian Distance '''
# non-negative value, lower the value -> higher the similarity
def get_eucledian_distance(job_embedding, resume_embedding, model):
  
  if model not in models:
    raise Exception('Model Not Found!')

  if model in ['fasttext', 'elmo']:
    return euclidean(job_embedding, resume_embedding)
  
  # GloVe, BERT, or GPT
  else:
    return np.linalg.norm(job_embedding - resume_embedding)
  

''' Pearson Correlation Coefficient'''
# -1 to 1. -1 perfect negative linear relationship, 0 none, 1 perfect positive
def get_pearson_coefficient(job_embedding, resume_embedding, model):
  
  if model not in models:
    raise Exception('Model Not Found!')

  if model == 'gpt':
    coeff, _ = pearsonr(job_embedding[0], resume_embedding[0])
    return coeff
  
  else:
    coeff, _ = pearsonr(job_embedding, resume_embedding)
    return coeff


''' Manhattan Distance '''
max_length = 10000

# non-negative. higher value -> less similar
def get_manhattan_distance(job_embedding, resume_embedding, model):
  
  if model not in models:
    raise Exception('Model Not Found!')

  if model in ['fasttext', 'glove', 'elmo']:
    return cityblock(job_embedding, resume_embedding)

  # BERT or GPT
  else:
    job_embedding = np.pad(job_embedding, (0, max_length - len(job_embedding)))
    resume_embedding = np.pad(resume_embedding, (0, max_length - len(resume_embedding)))
    return np.linalg.norm(job_embedding - resume_embedding, ord=1)
