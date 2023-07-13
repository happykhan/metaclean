import spacy
import logging 
import typer
import pandas as pd
import json 
import gzip
import os 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def create_dataframe(entero_file, data_table, training_num_records=100, testing_num_records=500, weight_field='Source Type', equal_sampling=True, random=False):
    """
    Fetches biosample data from two input files and returns a dictionary of training and testing data.

    Args:
        entero_file (str): Path to the enterobase file.
        data_table (str): Path to the all_attributes file.
        training_num_records (int): Number of training records to fetch.
        testing_num_records (int): Number of testing records to fetch.
        equal_sampling (bool, optional): Whether to sample each type equally. Defaults to True.
        weight_field (str, optional): Field to use for weighting. Defaults to 'Source Type'.

    Returns:
        tuple: A tuple of two dictionaries containing training and testing data.
    """    
    df = pd.read_csv(entero_file, compression='gzip', sep='\t')
    logging.info(f"Number of records in enterobase file: {len(df)}")
    # Filter where Sample ID is not null
    df = df[df['Sample ID'].notnull() & df['Source Type'].notnull()]
    if random:
        eb_records = df[['Sample ID', 'Source Type', 'Source Niche']].sample(n=(training_num_records+testing_num_records), random_state=7)
    else:
        if equal_sampling:
            df['freq'] = 1./df.groupby(weight_field)[weight_field].transform('count') # Sample each type equally 
        else: 
            # Subsample in the proportion of full df
            df['freq'] = df.groupby('Source Type')['Source Type'].transform('count')
        eb_records = df[['Sample ID', 'Source Type', 'Source Niche']].sample(n=(training_num_records+testing_num_records), random_state=7, weights=df.freq)
    eb_records_dict = {x['Sample ID']: x for x in  eb_records.to_dict('records') } 
    null_values = ['missing', 'not available', 'not collected', 'not applicable', 'not provided', 'not reported', 'unknown']
    # open all_attributes.csv.gz
    df = pd.read_csv(data_table, compression='gzip')
    logging.info(f"Number of records in all_attributes file: {len(df)}")
    # fetch dictionary of biosample details using field_list
    training_data_table = {} 
    testing_data_table = {}
    for i in range(len(df)):
        accession = df['Sample ID'][i]
        if eb_records_dict.get(accession):
            new_biosample = { 'Sample ID' : accession}
            new_biosample['EB Source Type'] = eb_records_dict[accession]['Source Type']
            new_biosample['EB Source Niche'] = eb_records_dict[accession]['Source Niche']
            field_list = df.columns.tolist()
            for field in field_list:
                if not pd.isna(df[field][i]):
                    if not str(df[field][i]).lower() in null_values:   
                        new_biosample[field] = str(df[field][i])
            new_biosample.pop('index', None)                        
            if len(training_data_table) < training_num_records:
                training_data_table[accession] = new_biosample
            else:
                testing_data_table[accession] = new_biosample
    return training_data_table, testing_data_table

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

def train_spacy_model(training_data_table):
    clean_training_data = {} 
    fields = fetch_field_list()
    for accession, biosample in training_data_table.items():
        source_type = biosample.pop('EB Source Type', None)
        source_niche = biosample.pop('EB Source Niche', None)
        sample_id = biosample.pop('Sample ID', None)
        #biosample_string = ','.join([str(key) + ':' + str(value) for key, value in biosample.items() if key in fields])
        # biosample_string = ','.join([str(value) for key, value in biosample.items() if key in fields])
        biosample_tags = [str(value) for key, value in biosample.items() if key in fields]
        if clean_training_data.get(source_type):
            clean_training_data[source_type] += biosample_tags
        else:
            clean_training_data[source_type] = biosample_tags
    # Remove duplicates
    for key, value in clean_training_data.items():
        clean_training_data[key] = list(set(value))
    # Train spacy model
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe(
        "text_categorizer",
        config={
            "data" : clean_training_data,
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "device": "gpu"
        }
    )
    return nlp

def create_or_load_tables(entero_file, data_table, training_num_records, testing_num_records, refresh=False):
    if not os.path.exists('training_table.gz') or refresh:
        logging.info("Creating training and testing data tables")
        training_data_table, testing_data_table = create_dataframe(entero_file, data_table, training_num_records=training_num_records, testing_num_records=testing_num_records)
        # write testing_data_table to gz file
        with gzip.open('training_table.gz', 'wt') as f:
            f.write(json.dumps(training_data_table))
        with gzip.open('testing_table.gz', 'wt') as f:
            f.write(json.dumps(testing_data_table))
    else:
        logging.info("Using existing training and testing data tables")
        training_data_table = json.loads(gzip.open('training_table.gz', 'rt').read())
        testing_data_table = json.loads(gzip.open('testing_table.gz', 'rt').read())    
    return training_data_table, testing_data_table

def load_or_train_model(training_data_table, refresh=False):
    if not os.path.exists('spacy_model') or refresh:
        logging.info("Building model from training data...")
        nlp = train_spacy_model(training_data_table)
        logging.info("Writing model to disk...")
        nlp.to_disk('spacy_model')
    else:
        logging.info("Loading model...")
        nlp = spacy.load("spacy_model")        
    return nlp

def test_model(testing_data_table, nlp, test_results='test_results.tsv'):
    # Test model
    logging.info("Beginning testing")
    pass_count = 0 
    total = 0
    fields = fetch_field_list()
    from statistics import mean
    with open(test_results, 'w') as f:
        f.write('Accesion\tEB Source Niche\tEB Source Type\tPredicted Source Type\tPerc\tPassed\tPrompt String\n')
        for accession, biosample in testing_data_table.items():
            total += 1
            filtered_biosample_string = ','.join([str(value) for key, value in biosample.items() if key in fields])
            
            biosample_tags = [str(value) for key, value in biosample.items() if key in fields]
            # filtered_biosample_string =  ','.join([str(key) + ':' + str(value) for key, value in biosample.items() if key in fields]) 
            result = []
            for tag in biosample_tags:
                doc = nlp(tag)
                top_hit = max(doc._.cats.items(), key=lambda x: x[1])
                result.append(top_hit[0])
            best_guess = 'Unknown'
            if result:
                best_guess = max(set(result), key=result.count)
            
            passed = 'NO'
            if best_guess == biosample['EB Source Type']:
                passed = 'YES'        
                pass_count += 1
            # doc = nlp(filtered_biosample_string)
            # top_hit = max(doc._.cats.items(), key=lambda x: x[1])
            # passed = 'NO'
            # if top_hit[0] == biosample['EB Source Type']:
            #     passed = 'YES'        
            #     pass_count += 1
            f.write('\t'.join([accession, biosample['EB Source Niche'], biosample['EB Source Type'], best_guess, str(round(top_hit[1], 3)), passed, filtered_biosample_string]) + '\n')
    logging.info(f"Passed {pass_count} out of {total} tests ({round(pass_count/total, 3) * 100} %)")

def compare_fields(fields="fields.csv"):
    # open fields.txt
    with open(fields, 'r') as f:
        all_field_list = f.readlines()[1:]
        # remove new line characters and remove empty strings
        nlp = spacy.load("en_core_web_lg")
        clean_field_list = ' '.join([field.split(",")[0].strip() for field in all_field_list  if field])
    return clean_field_list

def main(data_table:str="all_attributes.csv.gz", entero_file:str ="entero_all_7.7.23.tsv.gz", refresh:bool=False, training_num_records:int=100, testing_num_records:int=50):
    # compare_fields()
    training_data_table, testing_data_table = create_or_load_tables(entero_file, data_table, training_num_records, testing_num_records, refresh)
    original_training_data_table = training_data_table.copy()

    nlp = load_or_train_model(training_data_table, refresh)

    test_model(original_training_data_table, nlp, test_results='training_results.tsv')
    test_model(testing_data_table, nlp)

if __name__ == "__main__":
    typer.run(main)



