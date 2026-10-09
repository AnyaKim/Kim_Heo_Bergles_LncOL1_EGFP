#R script to grab files for index build and downstream kallisto bus
#Obtains:
#	cDNA_introns.fa
#	cDNA_tx_to_capture.txt
#	introns_tx_to_capture.txt
#	tr2g.txt

#Docs: https://rdrr.io/github/lambdamoses/BUStoolsR/man/get_velocity_files.html
print("Running... grabbing fastas for cDNA and introns...")

#If you need to link a specific R library
.libPaths("~/scratch/kallistobus/custom-R-lib1")

#Install and load necessary libraries
#BiocManager==3.10 for R 3.6
#updated to 3.22 for the current version
#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
##install.packages("BiocManager")
#BiocManager::install(version = "3.22")

##if (!require(devtools)) install.packages("devtools")
#default XML download fails for versions of R below 4.0, so download from source
#install.packages("https://cran.r-project.org/src/contrib/Archive/XML/XML_3.99-0.3.tar.gz", repos=NULL, type = "source")

##BiocManager::install("BiocGenerics")
##BiocManager::install("AnnotationHub")

#This is the only required package not in the singularity container
BiocManager::install("BSgenome.Mmusculus.UCSC.mm10")


##devtools::install_github("BUStools/BUSpaRse")
#had to download and reinstall current version of cli

library(BiocGenerics)
library(BUSpaRse)
library(AnnotationHub)
library(BSgenome.Mmusculus.UCSC.mm10)

print("Packages succesfully loaded.")

ah <- AnnotationHub() # Connect to the Annotation Hub to query.
record<-query(ah,pattern = c("Ensembl", "99", "Mus musculus", "EnsDb")) 
#Identify the AnnotationHub record associated with Mouse Ensembl v99


# Once we've identified the AnnotationHub record id `r names(record)` for Mouse Ensembl v99, we can use this to retrieve all of the annotation records for this build.
# Get mouse Ensembl 97 annotation
edb <- ah[[names(record)]] 

#L is dependent on sequencing chemistry
#10xv1: 98 nt, 10xv2: 98 nt, 10xv3: 91 nt
out_dir <- "~/scratch/kallistobus/required_files/"
get_velocity_files(edb, L = 91, Genome = BSgenome.Mmusculus.UCSC.mm10,out_path = out_dir, isoform_action = "separate")



