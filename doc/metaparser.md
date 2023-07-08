# About EnteroBase's metaparser

From https://enterobase.readthedocs.io/en/latest/pipelines/enterobase-pipelines.html and https://enterobase.readthedocs.io/en/latest/pipelines/backend-metaparser.html

## MetaParser
MetaParser implements - the automated downloading of all GenInfo Identifiers (GI numbers) in

NCBI Short Read Archives (SRAs) or complete or partial assemblies with the genus designation Salmonella, Escherichia / Shigella, Yersinia or Moraxella, and the corresponding metadata (via ENTREZ utilities)
parsing of the metadata into a consistent, EnteroBase format

**Assignment of Source Details into categories**

A 2-level source classification scheme (Table 1) was set up to cover a wide range of potential hosts or environments where the bacteria are mostly isolated from. In order to automatically assign genomic metadata from GenBank into Source Niche/Type categories, a 'Source Details' field was set up in EnteroBase to summarize all host-related biosample attributes for genomes in GenBank.

A subset of 3,546 distinct "Source Details" entries were manually assigned into Source Niches/Types in 2015 and used as ground truth to train a Native Bayesian classifier implemented in the Python NLTK library (Loper and Bird, 2002). During the training process, these manually curated data were randomly separated into a training and a test dataset with 2,000 and 1,546 entries, respectively.

The source classifier was trained using the training dataset and evaluated using the test dataset, achieving an accuracy of approximately 80%. Subsequently, the source classifier was trained again using all 3,546 entries and used to assign all GenBank entries into categories. Initially, this classifier performed well, but after 2 years, it encountered high frequencies of failed assignments in practice.

Therefore, its performance was re-evaluated in 2018 using an independent set of 3,000 manually curated entries. This time, the accuracy of the assignments dropped to 60%. Further evaluation identified the reduced accuracy as a result of a significant number of new words that were not recognized by the source classifier.

**Table 1: The Source Niche/Type classification scheme in EnteroBase**

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

Please note that you may need to adjust the table formatting according to the requirements of your markdown parser.