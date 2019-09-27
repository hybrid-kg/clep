#!/bin/bash

# Download Raw Data file
wget -O "GSE65682_RAW.tar" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE65682&format=file"
tar -zxvf "GSE65682_RAW.tar"
rm "GSE65682_RAW.tar"
gunzip -vf *.gz
echo "FileName" > "phenodata.txt"
ls *.CEL >> phenodata.txt

# Download the Platform design file
wget -O "GPL13667-15572.txt" "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&amp;is_datatable=true&amp;acc=GPL13667&amp;id=15572&amp;db=GeoDb_blob92"


# Download Miniml file
wget "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/miniml/GSE65682_family.xml.tgz"
tar -zxvf "GSE65682_family.xml.tgz" "GSE65682_family.xml"
rm "GSE65682_family.xml.tgz"

python3 preprocessing.py make-targets

Rscript affy_preprocessing.R

rm *.CEL

# Pre-Processing of Downloaded files
python3 preprocessing.py make-exp

# limma preprocessing
vp4p preprocessing limma --data exp.txt --design targets.txt --out limma.txt
# Z_Score preprocessing
vp4p preprocessing z-score --data exp.txt --design targets.txt --out z_score.txt
# limma data's threshold based vectorization
vp4p vectorization thresh2vec --data limma.txt --out thresh_limma.txt
# Z-Score data's threshold based vectorization
vp4p vectorization thresh2vec --data z_score.txt --out thresh_z_score.txt
