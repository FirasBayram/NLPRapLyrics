import pandas as pd
from nltk.corpus import stopwords
from nltk import regexp_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.collocations import BigramCollocationFinder
import operator

# The pattern we want to match
patn = '\w+'  #[a-zA-Z_][a-zA-Z0-9_]*
stw = ',\,-.!'
wordcount = {}
# Define the output dataframe
artists = ["Eminem", "Tupac"]
df = pd.DataFrame(columns=('artist', 'freq_words', 'Bigrams', 'BigramsPMI', 'BigramTTest'))
i = 0

# Fill the dataframe
for artist in artists:
    f = open(artist + '.txt', 'rb')
    raw_text = ""
    for sentence in f.readlines():
        this_sentence = sentence.decode('utf-8').lower()
        raw_text += this_sentence
    words= regexp_tokenize(raw_text, patn)
    # remove the stopwords
    words = [word for word in words if word not in stw]
    words = [word for word in words if word not in stopwords.words('english')]
    frequency = nltk.FreqDist(words)
    freq_words = sorted(frequency, key=frequency.__getitem__, reverse=True)[0:10]
    frequency.plot(20, cumulative=False)
    # Bigrams
    bigramFinder = BigramCollocationFinder.from_words(words)
    bigram_freq = sorted(bigramFinder.ngram_fd.items(), key=operator.itemgetter(1) , reverse=True) [:10]
    # Pointwise Mutual Information
    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramPMIFinder = BigramCollocationFinder.from_words(words)
    bigramPMIFinder.apply_freq_filter(35)
    bigramPMI = bigramPMIFinder.score_ngrams(bigrams.pmi)[:10]
    # Hypothesis Testing
    # t-test
    bigramtFinder = BigramCollocationFinder.from_words(words)
    bigramt = bigramtFinder.score_ngrams(bigrams.student_t)[:10]
    # Save the DataFrame
    df.loc[i] = (artist, freq_words, bigram_freq, bigramPMI ,bigramt)
    i += 1
    wcloud = WordCloud().generate_from_frequencies(frequency)
    plt.imshow(wcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
print(df)
df.to_csv("collocations2.csv", index=False)
