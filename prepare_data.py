import typer
import logging
import os
import gzip
import json
from create_dataset import create_dataframe
import pandas as pd
import re 
from sentence_transformers import SentenceTransformer
import spacy
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def create_or_load_tables(entero_file, data_table, training_num_records, refresh=False):
    """
    Creates or loads training and testing data tables from a given file.
    Args:
        entero_file (str): The path to the file containing the data.
        data_table (str): The name of the table to be created.
        training_num_records (int): The number of records to use for training.
        refresh (bool): Whether to refresh the data tables if they already exist.
    Returns:
        tuple: A tuple containing the training and testing data tables.
    Notes:
        This function first checks if the data tables already exist. If they do not exist or `refresh` is `True`, it creates the tables by calling the `create_dataframe` function. The resulting tables are then written to gzipped files. If the tables already exist and `refresh` is `False`, they are loaded from the gzipped files. The function returns a tuple containing the training and testing data tables.
    Examples:
        >>> entero_file = "path/to/data.csv"
        >>> data_table = "my_data_table"
        >>> training_num_records = 1000
        >>> create_or_load_tables(entero_file, data_table, training_num_records, refresh=True)
        ({...}, {...})
    """
    # TODO: Rewrite create_dataframe to return a dataframe instead of a dictionary
    if not os.path.exists('training_table.gz') or refresh:
        logging.info("Creating training and testing data tables")
        training_data_table = create_dataframe(entero_file, data_table, training_num_records=training_num_records, equal_sampling=True)
        # write testing_data_table to gz file
        with gzip.open('training_table.gz', 'wt') as f:
            f.write(json.dumps(training_data_table))
    else:
        logging.info("Using existing training and testing data tables")
        training_data_table = json.loads(gzip.open('training_table.gz', 'rt').read())
    return training_data_table

def clean_value(value):
    """
    Cleans a string by removing newlines, square brackets, and leading/trailing whitespace.
    Args:
        value (str): The string to be cleaned.
    Returns:
        str: The cleaned string.
    Notes:
        This function uses regular expressions to remove newlines and square brackets from the string. It also removes any leading or trailing whitespace using the `strip` method.
    Examples:
        >>> clean_value("[\n  test\n]")
        'test'
    """    
    value = re.sub(r'\n', '', value) 
    value = re.sub(r'\[|\]', "", value)
    value = value.strip()       
    return value

def clean_biosample(biosample):
    """
    Cleans a biosample by extracting relevant fields and concatenating them into a string.
    Args:
        biosample (pandas.DataFrame): The biosample to be cleaned.
    Returns:
        str: The cleaned biosample as a string.
    Notes:
        This function extracts the 'isolation_source', 'source_type', and 'host' fields from the biosample DataFrame and concatenates them into a string. The 'host' field is prefixed with 'animal_host'. The resulting string is returned.
    Examples:
        >>> biosample = pd.DataFrame({'isolation_source': 'soil', 'source_type': 'environment', 'host': np.nan})
        >>> clean_biosample(biosample)
        'soil environment'    
    """
    biosample_dict = biosample.to_dict()
    VALID_FIELDS = ['isolation_source'] # , 'source_type' ,'host']
    cleaned_biosample= []
    for field in VALID_FIELDS:
        if biosample_dict.get(field) and not pd.isna(biosample_dict[field]):
            cleaned_biosample.append(clean_value(biosample_dict[field]))
    cleaned_biosample_string = ' '.join(cleaned_biosample)
    print({k: v for k, v in biosample_dict.items() if not pd.isna(v)})
    print(cleaned_biosample_string)

    return cleaned_biosample_string 

def word2vec(text, nlp, wv):
    """    
    Calculates the word2vec vector representation of a given text.
    Args:
        text (str): The text to be processed.
        nlp (spacy.lang.en.English): The spaCy language model used to tokenize the text.
        wv (gensim.models.keyedvectors.Word2VecKeyedVectors): The pre-trained word2vec model used to calculate the vector representation of the text.
    Returns:
        numpy.ndarray: The word2vec vector representation of the text.
    Notes:
        This function filters out stop words and punctuation from the tokens, and lemmatizes the remaining tokens before calculating the vector representation.
    Examples:
        >>> text = "This is a test sentence."
        >>> nlp = spacy.load("en_core_web_sm")
        >>> wv = gensim.models.KeyedVectors.load_word2vec_format("path/to/word2vec.bin", binary=True)
        >>> word2vec(text, nlp, wv)
        array([ 0.01171875,  0.00341797, -0.00366211, ...])
    """
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    if not filtered_tokens:
        return np.zeros(300)
    else:
        return wv.get_mean_vector(filtered_tokens)

def tfidf_vectorize(text_list):
    """
    Calculates the TF-IDF vector representation of a list of texts.
    Args:
        text_list (list): A list of texts to be processed.
    Returns:
        list: The TF-IDF vector representation of the texts.
    Notes:
        This function uses the TfidfVectorizer class from scikit-learn to calculate the TF-IDF vector representation of the texts. The resulting vectors are converted to a dense matrix and returned as a list. 
    Examples:
        >>> text_list = ["This is a test sentence.", "This is another test sentence."]
        >>> tfidf_vectorize(text_list)
        [[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0]]    
    """
    vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_features=300,
                                max_df=0.7,
                                min_df=2,
                                ngram_range = (1,3),
                                stop_words = "english"

                            )
    vectors = vectorizer.fit_transform(text_list)    
    feature_names = vectorizer.get_feature_names_out()    
    dense = vectors.todense()
    denselist = dense.tolist()
    return denselist

def plot_groups(X, y, groups, out='plot_group.png'):
    """
    Plots a scatter plot of data points in two dimensions, where each point is colored according to its group assignment.
    Args:
        X (numpy.ndarray): The data points to plot.
        y (numpy.ndarray): The group assignments for each data point.
        groups (list): The list of unique group assignments.
        out (str): The filename to save the plot to.
    Returns:
        None
    Notes:
        This function loops over each group in the list of unique group assignments, and plots a scatter plot of the data points in two dimensions, where each point is colored according to its group assignment. The plot is saved to the specified filename using the `savefig` method, and the plot is cleared using the `clf` method.
    Examples:
        >>> X = np.random.rand(100, 2)
        >>> y = np.random.randint(0, 3, 100)
        >>> groups = [0, 1, 2]
        >>> plot_groups(X, y, groups, "scatter.png")
    """
    for group in groups:
        plt.scatter(X[y == group, 0], X[y == group, 1], label=group, alpha=0.4)
    # Make figure bigger
    plt.gcf().set_size_inches(8, 10)
    plt.legend()
    plt.savefig(out)    
    plt.clf()

def plot_cluster(df, yhat, c, cluster_out='cluster.png'):
    """
    Plots a bar chart of the counts of each EB source type in a given cluster.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        yhat (numpy.ndarray): The cluster assignments for each data point.
        c (int): The cluster to plot.
        cluster_out (str): The filename to save the plot to.
    Returns:
        None
    Notes:
        This function filters the DataFrame to only include data points in the specified cluster, and then plots a bar chart of the counts of each EB source type in the filtered DataFrame. The plot is saved to the specified filename.
    Examples:
        >>> df = pd.read_csv("path/to/data.csv")
        >>> yhat = kmeans.predict(df)
        >>> plot_cluster(df, yhat, 0, "cluster0.png")
    """
    df[yhat == c]['EB Source Type'].value_counts().plot(kind='bar', title="Cluster #{}".format(c))
    plt.savefig(cluster_out)    
    plt.clf()

def calculate_WSS(points, kmax=30):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

def clustering(df, method, out_dir, nlp, wss=False):
    """
    Performs clustering on a DataFrame using KMeans algorithm and plots the results.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        method (str): The name of the column in `df` to use for clustering.
        out_dir (str): The directory to save the output files to.
        nlp (spacy.Language): The Spacy language model to use for text preprocessing.
    Returns:
        pandas.DataFrame: The input DataFrame with a new column for the cluster assignments.
    Notes:
        This function performs clustering on the data in the specified column `method` of the input DataFrame `df` using KMeans algorithm. It then plots the results of the clustering using scatter plots and saves them to files in the specified output directory `out_dir`. The function also adds a new column to the input DataFrame `df` with the cluster assignments. The function returns the modified DataFrame.
    Examples:
        >>> df = pd.read_csv("path/to/data.csv")
        >>> nlp = spacy.load("en_core_web_sm")
        >>> clustering(df, "text", "output", nlp)
    """
    source_types = list(set(df['EB Source Type'].tolist()))
    X = normalize(np.stack(t for t in df[method]))
    logging.info("X (the document matrix) for {} has shape: {}".format(method, X.shape))
    logging.info("That means it has {} rows and {} columns".format(X.shape[0], X.shape[1]))
    # Check clustering with EB fields
    # cluster the attributes
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    logging.info("X2 shape is {}".format(X2.shape))
    plot_groups(X2, df['EB Source Type'], source_types, out=f'{out_dir}/{method}_plot_EB_groups.png')

    # First we fit the model...
    if wss:
        z = calculate_WSS(X)
        plt.plot(range(1,31), z)
        plt.xlabel('Number of clusters')
        plt.ylabel('WSS')
        plt.savefig(f'{out_dir}/{method}_wss.png')
        plt.clf()   
    CLUSTERS = 5 # OR len(source_types)
    k_means = KMeans(n_clusters=CLUSTERS, random_state=1)
    k_means.fit(X)
    # Let's take a look at the distribution across classes
    yhat = k_means.predict(X)
    # Clear the plot
    plt.hist(yhat, bins=range(CLUSTERS))
    # Write to plot to file
    plt.savefig(f'{out_dir}/{method}_cluster_distribution.png')    
    plt.clf()
    plot_groups(X2, yhat, range(CLUSTERS), out=f'{out_dir}/{method}_plot_kmeans_groups.png')
    for c in range(CLUSTERS):
        plot_cluster(df, yhat, c, cluster_out=f'{out_dir}/{method}_cluster_{c}.png')
    # Output df with new column for cluster
    df[f'{method}_cluster'] = yhat
    return df


def main(data_table:str="all_attributes.csv.gz", out_dir:str='clustering', entero_file:str ="entero_all_7.7.23.tsv.gz", refresh:bool=True, training_num_records:int=1000):
    """
    Prepare data for machine learning models.

    Args:
        data_table (str): Path to the data table file (default: "all_attributes.csv.gz").
        entero_file (str): Path to the enterobase table file (default: "entero_all_7.7.23.tsv.gz").
        refresh (bool): Whether to refresh the data table (default: False).
        training_num_records (int): Number of training records to use (default: 300).
        testing_num_records (int): Number of testing records to use (default: 300).

    Returns:
        None
    """    
    logging.info("Preparing data...")
    training_data_table = create_or_load_tables(entero_file, data_table, training_num_records, refresh)

    df = pd.DataFrame.from_dict(training_data_table.values())
    # Add a new column with the cleaned host string
    logging.info("Cleaning host string...")
    df['clean_host_string'] = df.apply(clean_biosample, axis=1)
    # Write to gz file
    df.to_csv('full_frame.gz', compression='gzip', index=False)
    print('\n'.join(df['clean_host_string'].tolist()))

    # Calculate vectorise with TF-IDF
    logging.info("Calculating vectors with TF-IDF...")
    df['tfidf'] = tfidf_vectorize(df['clean_host_string'])
                
    # Calculate vectors using GLOVE
    logging.info('Loading spacy model...')
    nlp = spacy.load("en_core_web_lg") 
    logging.info("Calculating vectors with GLOVE...")
    df['glove'] = df['clean_host_string'].apply(lambda text: nlp(text).vector)  

    # Calculate vectors using word2vec
    logging.info("Calculating vectors with word2vec...")
    wv = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
    df['word2vec'] = df['clean_host_string'].apply(lambda text: word2vec(text, nlp, wv))

    model_sent = SentenceTransformer('all-mpnet-base-v2')
    # Calculate vectors using MPNET
    logging.info("Calculating vectors with MPNET...")
    df['mpnet'] = df['clean_host_string'].apply(lambda text: model_sent.encode(text))

    # Output clustering
    logging.info("Clustering starts...")
    methods = ['tfidf', 'mpnet', 'glove', 'word2vec']
    # assign data of lists.  
    data = {'Name': methods,
         'Dimension': [len(df.tfidf[0]), len(df.mpnet[0]), len(df.word2vec[0]), len(df.glove[0])]}  
    # data to csv
    pd.DataFrame(data).to_csv('dimension.csv', index=False)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))    
    for method in methods:
        logging.info(f"Clustering {method}...")
        df = clustering(df, method, out_dir, nlp)
    df.to_csv('full_frame.gz', compression='gzip', index=False)
    logging.info("Clustering ends")      

if __name__ == "__main__":
    typer.run(main)

