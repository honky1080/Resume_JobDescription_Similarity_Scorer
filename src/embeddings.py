import numpy as np
from gensim.models import Word2Vec

def train_word2vec(tokenized_resumes,vector_size=100,window=5,min_count=2):
    print("Training Word2Vec Skip-gram model...")
    model = Word2Vec(
        sentences=tokenized_resumes,
        sg=1,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )
    return model

def vectorization(resume,model):
  resume_vector=[]
  for word in resume:
    if word in model.wv:
      resume_vector.append(model.wv[word])
  if len(resume_vector) == 0:
    return np.zeros(model.vector_size)
  resume_vector_mean=np.mean(resume_vector,axis=0)
  return resume_vector_mean

