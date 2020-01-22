from nltk.tag.stanford import StanfordNERTagger
import pandas  as pd
import nltk

# initialize NER tagger
sn = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
                       path_to_jar='stanford-ner.jar', encoding='utf8')
artists = ["Eminem", "Tupac"]
for artist in artists:
    f = open(artist+'.txt', 'rb')
    raw_text = ""
    for sentence in f.readlines():
        this_sentence = sentence.decode('utf-8')
        raw_text += this_sentence

    # tag named entities
    ner_tagged_sentences = [sn.tag(nltk.word_tokenize(raw_text))]

    # extract all named entities
    named_entities = []
    for sentence in ner_tagged_sentences:
        temp_entity_name = ''
        temp_named_entity = None
        for term, tag in sentence:
            if tag != 'O':
                temp_entity_name = ' '.join([temp_entity_name, term]).strip()
                temp_named_entity = (temp_entity_name, tag)
            else:
                if temp_named_entity:
                    named_entities.append(temp_named_entity)
                    temp_entity_name = ''
                    temp_named_entity = None

    # named_entities = list(set(named_entities))
    entity_frame = pd.DataFrame(named_entities,
                                columns=['Entity Name', 'Entity Type'])

    # view top entities and types
    top_entities = (entity_frame.groupby(by=['Entity Name', 'Entity Type'])
                    .size()
                    .sort_values(ascending=False)
                    .reset_index().rename(columns={0: 'Frequency'}))

    # view top entity types
    top_entities_ty = (entity_frame.groupby(by=['Entity Type'])
                    .size()
                    .sort_values(ascending=False)
                    .reset_index().rename(columns={0: 'Frequency'}))

    top_entities.to_csv(artist+"_TOP_EN_SF.csv", index=False)
    top_entities_ty.to_csv(artist+"_TOP_EN_TY_SF.csv", index=False)
