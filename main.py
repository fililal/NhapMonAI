import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from Memory_Base import Memory_Base
import process as app

model = app.process()
print(model.recommend(0))
model.change_rating(0, 32, 5)
app.save_data(model.rate, model.similarity_matrix)

