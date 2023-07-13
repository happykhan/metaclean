# Enterobase categories arent making sense
import typer
import pandas as pd
import logging
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from spacey_test import create_dataframe
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import os 
import gzip 
import json 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def fetch_field_list(fields="fields.csv"):
    """
    Fetches list of fields from fields.csv
    """
    # desired fields 
    wanted_fields = [
    'isolation_source',
    'attribute_package',
    'ontological term',
    'IFSAC+ Category',
    'source_type',
    'sample_type',
    'culture_collection',
    'package',
    'attribute package',
    'IFSAC+ category',
    'food_origin',
    'disease'
    ]
    # open fields.txt
    with open(fields, 'r') as f:
        all_field_list = f.readlines()[1:]
        # remove new line characters and remove empty strings
        clean_field_list = [field.split(",")[0].strip() for field in all_field_list  if field]
        field_list = [field.split(",")[0].strip() for field in clean_field_list  if field in wanted_fields or field.startswith("host")]
    return field_list

def print_comparison(a, b):
    # Euclidean "L2" distance
    distance = np.linalg.norm(a.vector - b.vector)
    # Cosine similarity
    similarity = a.similarity(b)
    print("-" * 80)
    print("A: {}\nB: {}\nDistance: {}\nSimilarity: {}".format(a, b, distance, similarity))


def vectorize(text, nlp):
    # Get the SpaCy vector -- turning off other processing to speed things up
    # vector = nlp(text).vector
    doc = nlp(text)
    vector = nlp(text)._.trf_data.tensors[0].reshape(-1, max(nlp(text)._.trf_data.tensors[0].shape)).squeeze().mean(axis=0)
    # vector = nlp(text)._.trf_data.tensors[-1]
    
    return vector

def plot_groups(X, y, groups, out='plot_group.png'):
    for group in groups:
        plt.scatter(X[y == group, 0], X[y == group, 1], label=group, alpha=0.4)
    # Make figure bigger
    plt.gcf().set_size_inches(10, 10)
    plt.legend()
    plt.savefig(out)    
    plt.clf()

def plot_cluster(df, yhat, c, cluster_out='cluster.png'):
    df[yhat == c]['EB Source Type'].value_counts().plot(kind='bar', title="Cluster #{}".format(c))
    plt.savefig(cluster_out)    
    plt.clf()

def main(entero_file:str ="entero_all_7.7.23.tsv.gz", out_dir:str='spacy', data_table:str="all_attributes.csv.gz", refresh:bool=False, training_num_records:int=5000, testing_num_records:int=50, model="en_core_web_trf"):
    if not os.path.exists('training_table.gz') or refresh:
        logging.info("Creating training and testing data tables...")
        # start with all_attributes.csv.gz
        training_data_table, testing_data_table = create_dataframe(entero_file, data_table, training_num_records=training_num_records, testing_num_records=testing_num_records, random=True)
        # write testing_data_table to gz file
        with gzip.open('training_table.gz', 'wt') as f:
            f.write(json.dumps(training_data_table))
        with gzip.open('testing_table.gz', 'wt') as f:
            f.write(json.dumps(testing_data_table))        
    else:
        logging.info("Using existing training and testing data tables...")
        training_data_table = json.loads(gzip.open('training_table.gz', 'rt').read())
        testing_data_table = json.loads(gzip.open('testing_table.gz', 'rt').read())    
    # Create a profile of attributes
    valid_fields = fetch_field_list()
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Number of records in all_attributes file: {len(training_data_table)}")
    logging.info(f"Loading model: {model}...")
    nlp = spacy.load(model)
    logging.info(f"Running through dataframe...")
#    prev = nlp('Something to compare')
    biosamples = [] 
    source_types = [] 
    for accession, biosample in training_data_table.items():
        vector_text = biosample.copy()
        vector_text.pop('Sample ID', None)
        source_types.append( vector_text.pop('EB Source Type', None))
        vector_text.pop('EB Source Niche', None)
        vector_text_string = ','.join([str(value) for key, value in vector_text.items() if key in valid_fields])
        biosample['vector_text'] = vector_text_string
        if vector_text_string != '':
            biosamples.append(biosample) 
        # thisone = nlp(vector_text_string)
        # print_comparison(thisone, prev)
        # prev = thisone
        # print(new_biosample)
    # vectorize the attributes
    df = pd.DataFrame.from_dict(biosamples)
    source_types = list(set(source_types))
    logging.info(f"Creating vectors...")
    X = normalize(np.stack(vectorize(t, nlp) for t in df['vector_text']))
    logging.info("X (the document matrix) has shape: {}".format(X.shape))
    logging.info("That means it has {} rows and {} columns".format(X.shape[0], X.shape[1]))
    # Check clustering with EB fields
    # cluster the attributes
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    logging.info("X2 shape is {}".format(X2.shape))
    plot_groups(X2, df['EB Source Type'], source_types, out=f'{out_dir}/plot_EB_groups.png')

    # First we fit the model...
    k_means = KMeans(n_clusters=len(source_types), random_state=1)
    k_means.fit(X)
    # Let's take a look at the distribution across classes
    yhat = k_means.predict(X)
    # Clear the plot
    plt.hist(yhat, bins=range(len(source_types)))
    # Write to plot to file
    plt.savefig(f'{out_dir}/cluster_distribution.png')    
    plt.clf()
    plot_groups(X2, yhat, range(len(source_types)), out=f'{out_dir}/plot_kmeans_groups.png')
    for c in range(len(source_types)):
        plot_cluster(df, yhat, c, cluster_out=f'{out_dir}/cluster_{c}.png')
    # Output df with new column for cluster
    df['cluster'] = yhat
    df.to_csv(f'{out_dir}/clustered.csv.gz', compression='gzip', index=False)

if __name__ == "__main__":
    typer.run(main)




