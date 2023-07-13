import spacy
from sense2vec import Sense2VecComponent

nlp = spacy.load("en_core_web_sm")
s2v = nlp.add_pipe("sense2vec")
s2v.from_disk("s2v_old")

doc = nlp("A sentence about natural language processing.")
assert doc[3:6].text == "natural language processing"
freq = doc[3:6]._.s2v_freq
vector = doc[3:6]._.s2v_vec
most_similar = doc[3:6]._.s2v_most_similar(3)

doc = nlp("A sentence about Facebook and Google.")
for ent in doc.ents:
    assert ent._.in_s2v
    most_similar = ent._.s2v_most_similar(3)