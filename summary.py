import pandas as pd
from nltk.corpus import stopwords
from nltk import regexp_tokenize
import textstat
import nltk
from textstat.textstat import legacy_round, textstatistics


# The pattern we want to match
patn = '\w+'
stw = ',\,-,!'

# Define the output dataframe
artists = ["Eminem", "Tupac"]
df = pd.DataFrame(columns=('artist', 'songs', 'words', 'num_sentences',
                           'words_per_song', 'vocabulary', 'sw_percentage',
                           'difficult_words', 'gf_readability', 'smog_readability'
                           , 'lr_readability'))
i = 0
sentences = 0

# Fill the dataframe
for artist in artists:
    f = open(artist + '.txt', 'rb')
    raw_text = ""
    for line in f.readlines():
        this_line = line.decode('utf-8')
        raw_text += this_line
    words = regexp_tokenize(raw_text, patn)
    print(len(words))
    # remove the stopwords
    words = [word for word in words if word not in stw]
    lyrics_no_sw = [word for word in words if word not in stopwords.words('english')]

    # Calculate the total number of words
    ttl_words = len(words)

    # Calculate the total number of sentences
    docReader = nltk.corpus.PlaintextCorpusReader('./', artist+ '.txt')
    sentences = len(docReader.sents())

    # Calculate the total number of difficult words
    diff_words_count = textstat.difficult_words(raw_text)

    # Calculate readability-- Gunning Fog
    dif_words = (diff_words_count / ttl_words * 100)
    gf_read = 0.4 * (float(ttl_words / sentences)+ dif_words)

    # Calculate readability-- SMOG
    poly_syl =0
    for word in words:
        syl_count = textstatistics().syllable_count(word)
        if syl_count >= 3:
            poly_syl += 1
    SMOG = (1.043 * (30 * (poly_syl / sentences)) ** 0.5) + 3.1291
    smog_read =  legacy_round(SMOG, 1)

    # Calculate readability-- Linsear Write
    cl_read = textstat.coleman_liau_index(raw_text)

    df.loc[i] = (artist, 0, ttl_words, sentences,
                 0, len(set(words)), round(100-(len(lyrics_no_sw)*100.0/ttl_words),2),
                 diff_words_count, gf_read, smog_read, cl_read)
    i += 1

df['songs'] = [304, 224]
df ['words_per_song'] = df['words']/df['songs']
print(df)
df.to_csv("summary.csv",index=False)