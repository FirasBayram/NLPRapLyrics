import pandas as pd
import nltk
import en_core_web_sm
from collections import Counter
from nltk.tag import StanfordNERTagger

# Import pretrained models
nlp = en_core_web_sm.load()
nlp.max_length = 2027221

# The pattern we want to match
patn = '\w+'
stw = ',\,-.!'
wordcount = {}
# Define the output dataframe
artists = ["Eminem", "Tupac"]
df = pd.DataFrame(columns=('artist', 'num_NE', 'labels', 'NER_spacy'))
i = 0

# Fill the dataframe
for artist in artists:
    f = open(artist + '.txt', 'rb')
    raw_text = ""
    for sentence in f.readlines():
        this_sentence = sentence.decode('utf-8')
        raw_text += this_sentence
    words= nltk.regexp_tokenize(raw_text, patn)
    # remove the stopwords
    words = [word for word in words if word not in stw]
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    article = nlp(str(words))
    labels = [x.label_ for x in article.ents]
    count = Counter(labels)
    items = [x.text for x in article.ents]
    df.loc[i] = (artist, len(article.ents), count, Counter(items).most_common(10))
    i += 1
df.to_csv("ENTITIES.csv", index=False)
