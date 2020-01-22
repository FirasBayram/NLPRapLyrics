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
vectorizer = TfidfVectorizer(stop_words = 'english', analyzer='word')
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
df = pd.DataFrame(denselist, columns=feature_names, index=['Eminem', "Tupac"])
print(df)
df.to_csv("tfidf.csv")

# Eminem top 1-gram terms
Em_df  = df.loc['Eminem']
Em_df = pd.DataFrame({'1_gram':Em_df.index, 'Score':Em_df.values})
Em_sorted =  Em_df.sort_values(['Score', '1_gram'], ascending=[0,1]).reset_index(drop=True)
print(Em_sorted['1_gram'][:10])
Em_df.to_csv("Eminem_1gram.csv")
Em_sorted.to_csv("Eminem_top_1gram.csv")
# Tupac top 1-gram terms
TP_df  = df.loc['Tupac']
TP_df = pd.DataFrame({'1_gram':TP_df.index, 'Score':TP_df.values})
TP_sorted =  TP_df.sort_values(['Score', '1_gram'], ascending=[0,1]).reset_index(drop=True)
print(TP_sorted['1_gram'][:10])
TP_df.to_csv("Tupac_1gram.csv")
TP_sorted.to_csv("Tupac_top_1gram.csv")
