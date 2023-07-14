# MetaClean

The purpose of `metaclean` is to use large language models to clean up semi-structured metadata 
and free text metadata presented in NCBI Biosample records. 

There's a long winded [explanation of the movitations here](doc/motivations.md). 

## Outline of (proposed) work

* [Matching EnteroBase's categorisation](doc/matching_enterobase.md)
* [Revisiting EnteroBase's categorisation](doc/revisiting_enterobase.md)
* [Standardising host names](doc/standardising_hostname.md)
* [Standardising geographic locations](doc/standardising_geoloc.md)


### Workflow 

* Fetch all Biosample data using `fetch_biosample.py` and validate 
* Create table of fields and create database tables `create_db.py`
* Create prompts for each record `create_prompt.py`


### Plan 

* Explore inherent clusters with tfidf
* Review text preprocessing. Tokenising may need to be adjusted.
* Explore inherent clusters from other vectorisation approaches - gpt, word2vec, glove, mpnet. 
* Tweak data processing based on above clusters - perhaps some existing EB ones are a little arbitrary. 
* Compare classification results with 'Logistic_Regression','Support_Vector_Machine', 'Random_Forest','Decision_Tree', - using each vectoriser above. (so that's 5 * 4 comparisons)
* Create training set of true labels.
* Repeat classification testing above.

https://derrickofori015.medium.com/gpt-3-vs-other-text-embeddings-techniques-for-text-classification-a-performance-evaluation-b3a3e6e84cb7
