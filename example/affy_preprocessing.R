# Reference: @Giuseppe Leite https://www.biorxiv.org/content/biorxiv/early/2018/02/26/271411.full.pdf

#######################################
###R Code Used to Preprocess Dataset###
#######################################

#Import affy libraries
library(affy)
library(simpleaffy)

#Read the targets file to load the cel files
celfiles <- read.affy(covdesc="targets.txt")

#Normalize the cel files
celfiles.rma <- rma(celfiles)

#Write the expression data to exp file
exp <- exprs(celfiles.rma)
write.table(exp, "exp.txt", sep="\t", quote=FALSE, col.names=NA)
