pip install gensim

import pandas as pd
import numpy as np
import nltk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

import os
os.chdir("/content/drive/MyDrive/Colab_data/Datasets")

# Use a pipeline as a high-level helper
from transformers import pipeline

classify = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# classify(["Biryani was good",
#           "Sambar was not so good.",
#           "Good Ambience.",
#           "There was a lot of noise in  the evening."])

dataset = pd.read_csv("Restaurant_Reviews.tsv", sep = '\t')
dataset

reviews = list(dataset['Review'])
sentiments = []
for r in tqdm(reviews):
  if classify(r)[0]['label'] == 'POSITIVE':
    sentiments.append(1)
  else:
    sentiments.append(0)

y = dataset['Liked']
accuracy_score(y, sentiments)