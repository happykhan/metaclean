import spacy
import pandas as pd

df1 = pd.read_csv("all_attributes.csv.gz", compression='gzip', header=0, sep=',', quotechar='"')
list(set(df1['isolation_source']))
# write to file
with open('isolation_source.txt', 'w') as f:
    for item in list(set(df1['isolation_source'])):
        f.write("%s\n" % item)
        

dataframe = 'full_frame.gz'
# Load the 'en_core_web_sm' model
nlp = spacy.load('en_core_web_lg')
df1 = pd.read_csv(dataframe, compression='gzip', header=0, sep=',', quotechar='"')
# run ner on the first 1000 rows of the dataframe
for i in range(1000):
    if not pd.isna(df1['clean_host_string'][i]):
        doc = nlp(df1['clean_host_string'][i])#
        print(df1['clean_host_string'][i])
        for ent in doc.ents:
            print(ent.text, ent.label_, ent.start_char, ent.end_char)

