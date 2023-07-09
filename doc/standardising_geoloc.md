# Standardising geographic locations
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