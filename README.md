# MetaClean

The purpose of `metaclean` is to use large language models to clean up semi-structured metadata 
and free text metadata presented in NCBI Biosample records. 

There's a long winded [explantations of the movitations here](doc/motivations.md). 

## Outline of (proposed) work

* [Matching EnteroBase's categorisation](doc/matching_enterobase.md)
* [Revisiting EnteroBase's categorisation](doc/revisiting_enterobase.md)
* [Standardising host names](doc/standardising_hostname.md)
* [Standardising geographic locations](doc/standardising_geoloc.md)


### Workflow 

* Fetch all Biosample data using `fetch_biosample.py` and validate 
* Create table of fields and create database tables `create_db.py`
* Create prompts for each record `create_prompt.py`


