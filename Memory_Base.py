import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class Memory_Base(object):
  def __init__(self, rate_data, Nuser, Nmovie, similarity_matrix = None):
    self.rate = rate_data
    self.Nuser = int(Nuser)
    self.Nmovie = int(Nmovie)
    if similarity_matrix.any() == None:
      self.fit()
    else:
      self.similarity_matrix = similarity_matrix
      self.normalize()
      self.making_sparse_tensor()

  def normalize(self):
    rate_copy = self.rate.copy()
    mu = np.zeros((self.Nuser, ))
    userCol = self.rate[:, 0]
    for n in range(self.Nuser):
      idx = np.where(userCol == n)[0].astype(np.int32)
      if idx.shape[0] == 0:
        continue
      item_idx = self.rate[idx, 1]
      ratings = self.rate[idx, 2]
      m = np.mean(ratings)
      mu[n] = m
      rate_copy[idx, 2] = ratings - mu[n]
    self.rate_normalize = rate_copy

  def making_sparse_tensor(self):
    self.rate_sparse_tensor = tf.sparse.SparseTensor(indices=self.rate_normalize[:, :2],
                                        values=self.rate_normalize[:, 2],
                                        dense_shape=[self.Nuser, self.Nmovie])
    
  def similarity(rate):
    sparse_matrix = sparse.coo_matrix((rate[:, 2], 
                                      (rate[:, 0], rate[:, 1])),
                                      shape = (Nuser, Nmovie))
    sparse_matrix = sparse_matrix.tocsr()
    S = cosine_similarity(sparse_matrix, sparse_matrix)
    for n in range(Nuser):
      S[n, n] = 1
    return tf.Variable(S)

  def similarity_user(self, userA, userB):
    cos_sim = np.dot(userA, userB)/(np.linalg.norm(userA)*np.linalg.norm(userB))
    return cos_sim
  
  def similarity_step_by_step(self):
    U_matrix = tf.sparse.to_dense(self.rate_sparse_tensor).numpy()
    similarity_matrix = np.zeros(shape=(self.Nuser, self.Nuser), dtype=np.float64)
    for userA in range(self.Nuser):
      for userB in range(self.Nuser):
        if userA == userB:
          similarity_matrix[userA, userB] = 1
        if userA < userB:
          similarity_matrix[userA, userB] = self.similarity_user(U_matrix[userA], U_matrix[userB])
          similarity_matrix[userB, userA] = self.similarity_matrix[userA, userB]

    self.similarity_matrix = similarity_matrix

  def predict(self, user, movie, k = 10):
    # index = np.where(self.rate[:, 1] == movie)[0].astype(np.int16)
    # rating_movie = self.rate[index]
    # user_rate = rating_movie[:, 0].astype(np.int16)
    # movie_rating = rating_movie[:, 2]
    # sim_user = self.similarity_matrix[user_rate][:, user]
    # select_user = sim_user.argsort()[-k:]
    # a = np.matmul(movie_rating[select_user], sim_user[select_user])
    # b = np.sum(abs(sim_user[select_user]))
    # return (a/b)
    index = tf.where(self.rate[:, 1] == movie).numpy()
    rating_movie = self.rate[index][:, 0]
    user_rate = rating_movie[:, 0].astype(np.int16)
    movie_rating = rating_movie[:, 2]
    sim_user = self.similarity_matrix[user_rate][:, user]
    select_user = sim_user.argsort()[-k:]
    a = np.matmul(movie_rating[select_user], sim_user[select_user])
    b = np.sum(abs(sim_user[select_user]))
    return (a / b)

  def recommend(self, user, need = 10):
    index = tf.where(self.rate[:, 0] == user).numpy()
    movie_user_rated = self.rate[index][:, 0][:, 1].astype(np.int16)
    Npred = self.Nmovie - index.shape[0]
    red = np.zeros(shape=(0, 2))
    for n in range(self.Nmovie):
      if n not in movie_user_rated:
        predict_value = self.predict(user, n)
        if np.isnan(predict_value):
          continue
        red = np.concatenate((red, np.array([[n, predict_value]])))
    needTake = red[:, 1].argsort()[-need:]
    return red[needTake][:, 0].astype(np.int16).tolist()

  def add_new_rating(self ,user, movie, rating):
    self.rate = np.concatenate((self.rate, np.array([[user, movie, rating]])))
    self.fit()

  def change_rating(self, user, movie, rating):
    index1 = np.where(self.rate[:, 0] == user)[0].astype(int)
    index2 = np.where(self.rate[index1, 1] == movie)[0].astype(int)
    self.rate[index2, 2] = rating
    self.fit()
  
  def fit(self):
    self.normalize()
    self.making_sparse_tensor()
    self.rate_sparse_tensor = tf.sparse.reorder(self.rate_sparse_tensor)
    self.similarity_step_by_step()