from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Define the documents and corpus
documentA= ''
f1 =  open('Eminem.txt', 'rb')
for sentence in f1.readlines():
    this_sentence = sentence.decode('utf-8').lower()
    documentA += this_sentence
documentB= ''
f2 =  open('Tupac.txt', 'rb')
for sentence in f2.readlines():
    this_sentence = sentence.decode('utf-8').lower()
    documentB += this_sentence

# Define the vectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', analyzer='word',ngram_range=(2,2))
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()

# Get the top 10 tfidf scores
feature_array = np.array(feature_names)
tfidf_sorting = np.argsort(vectors.toarray()).flatten()[::-1]
n = 10
top_n = feature_array[tfidf_sorting][:n]
print(top_n)

# Save the result in a dataframe
df = pd.DataFrame(denselist, columns=feature_names, index=  ['Eminem', 'Tupac'])
df.to_csv("tfidf_2grams.csv")


# Eminem top 2-grams terms
Em_df  = df.loc['Eminem']
Em_df = pd.DataFrame({'2_grams':Em_df.index, 'Score':Em_df.values})
Em_sorted =  Em_df.sort_values(['Score', '2_grams'], ascending=[0,1])
print(Em_sorted[:10])
Em_df.to_csv("Eminem_2grams.csv")
Em_sorted.to_csv("Eminem_top_2gram.csv")


# Tupac top 2-grams terms
TP_df  = df.loc['Tupac']
TP_df = pd.DataFrame({'2_grams':TP_df.index, 'Score':TP_df.values})
TP_sorted =  TP_df.sort_values(['Score', '2_grams'], ascending=[0,1])
print(TP_sorted[:10])
TP_df.to_csv("Tupac_2grams.csv")
TP_sorted.to_csv("Tupac_top_2gram.csv")
