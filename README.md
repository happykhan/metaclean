# MetaClean

The purpose of `metaclean` is to use large language models to clean up semi-structured metadata 
and free text metadata presented in NCBI Biosample records. 

## Introduction

I originally encountered this problem working on [EnteroBase](https://enterobase.warwick.ac.uk/). 
From the group's experience with MLST databases, it was clear that sequence data needed systematic 
and curated metadata to answer questions around source attribution, and spatial and temporal trends
of bacterial pathogens. We spent some time discussing how detailed categories and free text could be 
classified and much of these ideas were implemed by my colleague Zhemin Zhou in [EnteroBase's Metaparser](doc/metaparser.md). 

Metaparse is a 2-level source classification scheme covered various hosts and environments. Genomic metadata 
from GenBank was assigned using the 'Source Details' field in EnteroBase. A subset of 3,546 entries trained 
a Native Bayesian classifier with 80% accuracy.

The process was labourious, I remember spending several weeks building the training dataset. And aftwared, 
we would often manually curate the results as we spotted errors.

We were not the only ones, there have been other efforts to address this problem, such as [LexMappr](https://www.cineca-project.eu/blog-all/lexmapr-a-rule-based-text-mining-tool-for-ontology-term-mapping-and-classification)([Github](https://github.com/Public-Health-Bioinformatics/LexMapr)).

Here is the problem. In most projects I have been involved with we want to categorise samples in terms of:

* Host / Isolation source
* Geographic location
* & Time

This is not always easy to retrieve from the wealth of publicly available data. 
Let's take this is a [simple example of a Biosample](https://www.ncbi.nlm.nih.gov/biosample/SAMN15894674). 

| Identifiers    |  Values                            |
|----------------|------------------------------------|
| BioSample      | SAMN15894674                       |
| Sample name    | FSIS22029019                       |
| SRA            | SRS7251715                         |
| Organism       |                                    |
| - Genus        | Enterobacteriaceae                  |
| - Species      | Salmonella                         |
| - Subspecies   | Salmonella enterica                |
| - Subspecies 2 | Salmonella enterica subsp. enterica |
| - Serovar      | Eko                                |
| Package        |                                    |
| - Category     | Pathogen: environmental/food/other |
| - Version      | 1.0                                |
| Attributes     |                                    |
| - Strain       | FSIS22029019                       |
| - Collected by | USDA-FSIS                          |
| - Collection date | 2020                           |
| - Isolation source | Product-Raw-Intact-Pork         |
| - Geographic location | USA:NC                    |
| - Latitude and longitude | missing                |
| - Serovar      | Salmonella enterica subsp. enterica serovar Eko |
| - Subspecies   | enterica                           |
| BioProject     | PRJNA242847 Salmonella enterica    |
| Submission     | USDA-FSIS; 2020-08-24              |

This is a very orderly example, but the details are not what quite what I would like for general classification.
For instance, a human being can infer. 

* The original host animal is Swine - we could even assume domestic pig. 
* Where this sample was taken in the food production chain is unclear, but we can guess this sample is not clinically related. 
* Even though the collection date only mentions the year, the sample was probably collected before the data was submitted (24-08-2020).
* The sample was (probably) collected in North Carolina, USA, North America. 

But simple progammatic text search will not help us with automating these inferences. For instance, Swine and Pork are obviously 
not lexographically similar, making it difficult to programmatically collate similar samples. Even though the metadata above is 
very well strutured, we still need to standardise this information for our purposes. 

With recent improvements in large language models, it seems possible that these inferences could be automated. And the aim here, is
to attempt to automate these inferences (using ChatGPT) and assess performance. [ChatGPT](https://chat.openai.com/) descrives itself as:

```
ChatGPT, developed by OpenAI, is an advanced language model based on the GPT (Generative Pre-trained Transformer) architecture. 
It has been trained on a vast amount of text data from the internet and is designed to generate human-like responses in a conversational manner. 
```

## Outline of (proposed) work

### Matching EnteroBase's categorisation

This new fangled technology shouls surely match the categorisation by [EnteroBase](doc/metaparser.md). Given some samples with the existing 
categories from EnteroBase, as the table below shows, can 

| Source Niche   | Source Type                                      | Examples of Source Details    |
| -------------- | ------------------------------------------------ | ----------------------------- |
| Aquatic        | Fish; Marine Mammal; Shellfish                   | Tuna, lobster                 |
| Companion Animal | Canine; Feline                                  | Cat, dog                      |
| Environment    | Air; Plant; Soil/Dust; Water                     | River, tree, soil             |
| Feed           | Animal Feed; Meat                                | Dog treat, fishmeal           |
| Food           | Composite Food; Dairy; Fish; Meat; Shellfish     | Milk, salami, ready-to-eat food |
| Human          | Human                                            | Patient, biopsy               |
| Laboratory     | Laboratory                                       | Reference strain, serial passage |
| Livestock      | Bovine; Camelid; Equine; Ovine; Swine            | Horse, calf                   |
| Poultry        | Avian                                            | Turkey, chicken               |
| Wild Animal    | Amphibian; Avian; Bat; Bovine; Camelid; Canine; Deer; Equine; Feline; Invertebrates; Marsupial; Other Mammal; Ovine; Primate; Reptile; Rodent; Swine | Flamingo, frog, python, Spider |
| ND             | ND                                               | ND                            |

### Comparing EnteroBase's categorisation



### Standardising host names
Host names (i.e. the host organism) can often be written as scientific name or common names. So given either 
a scientific name or common name can it be made consistent like the table below?

| Number | Common Name              | Scientific Name         |
|--------|--------------------------|-------------------------|
| 37     | Virginia opossum        | Didelphis virginiana    |
| 38     | Mountain horned dragon  | Acanthosaura capra      |
| 39     | Cuttlefish              | Sepia officinalis       |
| 40     | Common musk turtle      | Sternotherus odoratus   |
| 41     | Human (female)          | Homo sapiens            |
| 42     | Water                    | N/A                     |
| 43     | Hedgehog                | Erinaceus europaeus     |
| 44     | Eurasian wigeon         | Mareca penelope         |
| 45     | Domestic goat           | Capra hircus            |
| 46     | Turkey                   | Meleagris gallopavo     |
| 47     | Domestic pig             | Sus scrofa domesticus   |
| 48     | Elephant seal            | Mirounga sp.            |
| 49     | Spinach                  | Spinacia oleracea       |

### Standardising geographic locations
Biosample records do not have the same level of completeness, they may only mention the city or the country of collection. In truth, having
all geographic fields would be best to make samples directly comparable. 

E.g. 

* Sample 1 location. "Paris" 
* Sample 2 location. "France"
* Sample 3 location. "London"

If we wanted to include all samples from France, we would want include both samples 1 and 2. Hence, the desired output would be 
for every sample to have:

* Continent
* Country
* State/Province/County
* City
* Approxiamate latitude 
* Approxiamate longitude 

Is this something that could be backfilled?

### 





### Workflow 

* Fetch all Biosample data using `fetch_biosample.py` and validate 
* Create table of fields and create database tables `create_db.py`
* Create prompts for each record `create_prompt.py`


