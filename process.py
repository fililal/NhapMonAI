import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from Memory_Base import Memory_Base

def process():
  similarity = pd.read_csv('Similarity_Matrix.csv').to_numpy()
  rate = pd.read_csv
  r_cols = ['user_id', 'movie_id', 'rating']
  ratings_base = pd.read_csv('rate.csv', names = r_cols, encoding='latin-1')
  rate = ratings_base.to_numpy(dtype=np.float32)
  user_movie = pd.read_csv('Data.csv', names=['NumberOfUser', 'NumberOfMovie']).to_numpy()
  Nuser = user_movie[:, 0]
  Nmovie = user_movie[:, 1]
  return Memory_Base(rate, Nuser, Nmovie, similarity)

def save_data(rate, similarity_matrix):
  r_cols = ['user_id', 'movie_id', 'rating']
  df = pd.DataFrame(rate.astype(np.int32), columns=r_cols)
  df.to_csv('rate.csv', header=None)
  df = pd.DataFrame(similarity_matrix)
  df.to_csv('Similarity_Matrix.csv')