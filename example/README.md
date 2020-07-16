Example
=======
This README will walk you through a simple usage of CLEP.

Alternatively [run.sh](run.sh) can be executed to run the example for you

Downloading Data
-------------------
1. Download the data, [GSE65682](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE65682&format=file)
2. Unzip the file and and its constituent CEL files
3. Download and unzip the Platform Design file [GPL13667-15572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&amp;is_datatable=true&amp;acc=GPL13667&amp;id=15572&amp;db=GeoDb_blob92) and the MINIML xml file [GSE65682_family.xml](ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/miniml/GSE65682_family.xml.tgz)

Pre-Processing the Data
--------------------------
1. Add all the CEL filenames to a tab separated file with the heading as "FileName"
2. Run the preprocessing python script using the command,

    `python3 preprocessing.py make-targets`
3. Then process the expression matrix using the R affy_preprocessing script using the command,

    `Rscript affy_preprocessing.R`
4. Generate the expression file of the correct format using the python script,

    `python3 preprocessing.py make-exp`

Single Sample Scoring
------------------------
- For limma based Single Sample Scoring, run,

    `clepp preprocessing limma --data exp.txt --design targets.txt --out limma.txt`
- For Z-Score based Single Sample Scoring, run,

    `clepp preprocessing z-score --data exp.txt --design targets.txt --out z_score.txt`


