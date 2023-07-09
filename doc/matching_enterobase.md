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

One approach is to give  (without any specific training)

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

