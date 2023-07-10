# Matching EnteroBase's categorisation

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

## A cold start

One approach is to give the GPT the NCBI biosample metadata and ask it to classify into one of the EnteroBase fields (without any specific training). This will allow us to assess GPT's 
success in replicating EnteroBase, without having to develop our own curated test set. Keep in mind, the EnteroBase classifications are not perfect so this is not evidence that GPT can or can not 
provide true classification. 

A list of the possible values for source type in EnteroBase.
```
Fish, Marine Mammal, Shellfish, Canine, Feline, Air, Plant, Soil/Dust, Water, Animal Feed, Meat, Composite Food, Dairy, Fish, Meat, Shellfish, Human, Laboratory, Bovine, Camelid, Equine, Ovine, Swine, Avian
Amphibian, Avian, Bat, Bovine, Camelid, Canine, Deer, Equine, Feline, Invertebrates, Marsupial, Other Mammal, Ovine, Primate, Reptile, Rodent, Swine, Not determined
```

A possible example prompt: 
```
Given the ncbi biosample information. 

isolation source	river water
source type	environmental
collection date	2008
geographic location	Mexico
latitude and longitude	missing
strain	CFSAN039526
collected by	CIAD-UAS
serovar	Pomona
sub species	enterica
project name	GenomeTrakr
sequenced by	FDA Center for Food Safety and Applied Nutrition
attribute_package	environmental/food/other
ontological term	river water:envo_01000599
IFSAC+ Category	environmental-water

Assign the record to one of the following host categories.

Fish, Marine Mammal, Shellfish, Canine, Feline, Air, Plant, Soil/Dust, Water, Animal Feed, Meat, Composite Food, Dairy, Fish, Meat, Shellfish, Human, Laboratory, Bovine, Camelid, Equine, Ovine, Swine, Avian
Amphibian, Avian, Bat, Bovine, Camelid, Canine, Deer, Equine, Feline, Invertebrates, Marsupial, Other Mammal, Ovine, Primate, Reptile, Rodent, Swine, Not determined

```

### First attempt

Using the script in `matching_enterobase.py`
I created some prompts to assign a category for a few biosamples and compared it back with the original classification from Enterobase. The prompts were something
like:

```
Given the ncbi biosample information. Sample
ID:SAMN03083832,collected_by:FDA,isolation_source:rapeseed meal
canola,attribute_package:environmental/food/other,ontological term:rapeseed
meal:FOODON_03310043|canola meal:FOODON_00002694,IFSAC+
Category:seeds,source_type:food. Assign the record to one of the following host
categories: Camelid,Not determined,Equine,Feline,Marine Mammal,Dairy,Primate,Soi
l/Dust,Meat,Human,Air,Swine,Deer,Amphibian,Reptile,Bat,Bovine,Shellfish,Marsupia
l,Invertebrates,Ovine,Water,Canine,Other Mammal,Animal
Feed,Fish,Rodent,Avian,Plant,Laboratory,Composite Food. Only reply with the
category. Use Not determined if not sure.
```

Here are the results for a random selection of biosamples:

Here's the updated table in Markdown format:


| Sample_ID    | EB_Source_type            | GPT_Source_type       | Result                              |
|--------------|---------------------------|-----------------------|-------------------------------------|
| SAMN02352711 | Plant                     | Plant                 | x                                   |
| SAMN27768017 | Bovine                    | Animal                | x                                   |
| SAMN10359333 | Avian                     | Avian                 | x                                   |
| SAMN31142663 | Water/River               | Not determined        | x                                   |
| SAMN24442301 | Bovine                    | Meat                  | x                                   |
| SAMN06481922 | Human                     | Human                 | x                                   |
| SAMN25002870 | Plant                     | Food                  | x                                   |
| SAMN14861190 | Human                     | Human                 | x                                   |
| SAMN13414191 | Human                     | Human                 | x                                   |
| SAMN12238825 | ND/Others                 | Food                  | fail                                |
| SAMN17216187 | Human                     | Human                 | x                                   |
| SAMN33744402 | Poultry                   | Avian                 | x                                   |
| SAMN08624184 | Poultry                   | Meat                  | x                                   |
| SAMN02253015 | Fish                      | Fish                  | x                                   |
| SAMN09431297 | Human                     | Human                 | x                                   |
| SAMN19548558 | Human                     | The record should be assigned to the "Human" host category. | x                  |
| SAMN09434936 | Human                     | Human                 | x                                   |
| SAMN05301281 | Swine                     | Swine                 | x                                   |
| SAMN10064125 | Swine                     | Swine                 | x                                   |
| SAMN27754168 | Human                     | Human                 | x                                   |
| SAMN27024394 | ND/Others                 | Animal                | x                                   |
| SAMN03894239 | Poultry                   | The record can be assigned to the "Avian" category. | x                           |
| SAMN06183468 | Poultry                   | Meat                  | x                                   |
| SAMN15216731 | Weasel/Badger (Mustelid)  | Not determined        | eb_error                            |
| SAMN02403350 | Bovine                    | Bovine                | x                                   |
| SAMN16261443 | Bovine                    | Bovine                | x                                   |
| SAMN30955101 | Poultry                   | Poultry               | fail                                |
| SAMN09522017 | Human                     | Human                 | x                                   |
| SAMN09203646 | Human                     | Human                 | x                                   |
| SAMN12212484 | Swine                     | Swine                 | x                                   |
| SAMN07514578 | Human                     | Human                 | x                                   |
| SAMN09916170 | Poultry                   | Avian                 | x                                   |
| SAMN08929367 | Poultry                   | Avian                 | x                                   |
| SAMN20960601 | Human                     | Human                 | x                                   |
| SAMN24657383 | Avian                     | Avian                 | x                                   |
| SAMN03479563 | Human                     | Human                 | x                                   |
| SAMN10080287 | Human                     | Human                 | x                                   |
| SAMN03577268 | Poultry                   | The record should be assigned to the category "Meat". | x                          |
| SAMN08767345 | Poultry                   | Avian                 | x                                   |
| SAMN09475168 | Human                     | Human                 | x                                   |
| SAMN28600665 | Bovine                    | Bovine                | x                                   |
| SAMN19605052 | Poultry                   | Avian                 | x                                   |
| SAMN06278516 | Human                     | Human                 | x                                   |
| SAMN09371598 | Human                     | Human                 | x                                   |
| SAMN09405169 | Human                     | Not determined        | eb_error                            |
| SAMN04601104 | Human                     | Human                 | x                                   |
| SAMN08161536 | Human                     | Human                 | x                                   |
| SAMN03083832 | Plant                     | AnimalFeed            | eb_error                            |
| SAMN09100764 | Human                     | Human                 | x                                   |
| SAMN09770906 | ND/Others                 | Poultry               | fail                                |
| SAMN11474784 | Human                     | Human                 | x                                   |
| SAMN16481075 | ND/Others                 | Avian                 | eb_error                            |
| SAMN03104699 | Human                     | Human                 | x                                   |
| SAMN07173395 | Human                     | Human                 | x                                   |
| SAMN12262008 | Human                     | Human                 | x                                   |
| SAMN07501475 | Poultry                   | The record can be assigned to the "Poultry" category. | fail                       |
| SAMN08294319 | Avian                     | Meat                  | x                                   |
| SAMN08567762 | Human                     | Human                 | x                                   |
| SAMN18675266 | Human                     | The record should be assigned to the "Human" category. | x                          |
| SAMN02698346 | Fish                      | Shellfish             | eb_error                            |
| SAMN18820663 | Human                     | Human                 | x                                   |


In the results column, the X means that GPT passed. The eb_error means that there the original EnteroBase assignment is actually wrong. Fail in this case is where the category was incorrect. In every fail, GPT correctly assigned the category Poultry or Food to the record, but these were not in the list of valid categories. 

In some cases, GPT also added extra text "the record should be ..." when told only to return the category.

So in this first attempt, GPT can correctly classify all the records, better than the EnteroBase. e.g. SAMN15216731 is assigned in EnteroBase as Weasel/Badger (Mustelid), but the metadata only metions "Small intestine" with no other host information. GPT correctly assigns this as "Not determined". However, GPT introduces its own (correct) categories (like Poultry) when it shouldn't.  