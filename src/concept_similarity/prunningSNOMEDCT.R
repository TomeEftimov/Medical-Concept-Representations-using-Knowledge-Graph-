#SNOMED CT prunning script


library(data.table)
library(fastmatch)

#Read the UMLS semantic groups file

semantic_groups<-fread(file="/SNOMED prunning/SemGroups_2018.txt",sep="|",header=FALSE)

# Extract only the semantic types related to clinical concepts inlcluding the semantic groups (CHEM, DISO, PROC, ANAT)
chem<-semantic_groups[semantic_groups$V1=="CHEM",]
diso<-semantic_groups[semantic_groups$V1=="DISO",]
proc<-semantic_groups[semantic_groups$V1=="PROC",]
anat<-semantic_groups[semantic_groups$V1=="ANAT",]

#Exlude some of the semantic types that belong to the selected semantic groups, but are not realted to clincial concepts

 anat<-anat[-c(7,8,9),]
 chem<-chem[-c(4,5,6,7,9),]
 diso<-diso[-c(6),]


#Extract the identifiers of the semantic types
TUI_chem<-as.character(chem$V3)
TUI_devi<-as.character(devi$V3)
TUI_diso<-as.character(diso$V3)
TUI_proc<-as.character(proc$V3)
TUI_anat<-as.character(anat$V3)
TUI_phys<-as.character(phys$V3)

#Read the MRSTY.RRF data 
#Please insert the path to your MRST file

mrsty_data<-fread(file="",sep="|",header=FALSE,verbose=TRUE)

mrsty_data_slected_semantic_groups<-mrsty_data[mrsty_data$V2 %in% c(TUI_chem,TUI_diso,TUI_proc,TUI_anat),]

TUIs<-as.character(mrsty_data_slected_semantic_groups$V2)
CUIs<-as.character(mrsty_data_slected_semantic_groups$V1)

mrsty_data_chem<-mrsty_data[mrsty_data$V2 %in% TUI_chem,]
mrsty_data_diso<-mrsty_data[mrsty_data$V2 %in% TUI_diso,]
mrsty_data_proc<-mrsty_data[mrsty_data$V2 %in% TUI_proc,]
mrsty_data_anat<-mrsty_data[mrsty_data$V2 %in% TUI_anat,]

CUIs_chem<-unique(mrsty_data_chem$V1)
CUIs_diso<-unique(mrsty_data_diso$V1)
CUIs_proc<-unique(mrsty_data_proc$V1)
CUIs_anat<-unique(mrsty_data_anat$V1)



#Extract all unique identifiers (i.e. CUIs) for each semantic group
CUIs_chem<-unique(CUIs_chem)
CUIs_diso<-unique(CUIs_diso)
CUIs_proc<-unique(CUIs_proc)
CUIs_anat<-unique(CUIs_anat)


#read the MRCONSO.RRF data
#Please insert the path to your MRCONSO file
mrconso_data<-fread(file="",sep="|",header=FALSE,verbose=TRUE)


#Extract all English concepts 
mrconso_data_eng<-mrconso_data[mrconso_data$V2=="ENG",]
mrconso_data_semantic_groups<-mrconso_data_eng[mrconso_data_eng$V1 %in% unique(CUIs),]
mrconso_data_snomed<-mrconso_data_semantic_groups[startsWith(as.character(mrconso_data_semantic_groups$V12),"SNOMEDCT"),]

#Spit the concepts per semantic groups
mrconso_data_chem<-mrconso_data_eng[mrconso_data_eng$V1 %in% unique(CUIs_chem),]
mrconso_data_diso<-mrconso_data_eng[mrconso_data_eng$V1 %in% unique(CUIs_diso),]
mrconso_data_proc<-mrconso_data_eng[mrconso_data_eng$V1 %in% unique(CUIs_proc),]
mrconso_data_anat<-mrconso_data_eng[mrconso_data_eng$V1 %in% unique(CUIs_anat),]

#Extract only those concepts that come from SNOMED CT
mrconso_snomed_chem<-mrconso_data_chem[startsWith(as.character(mrconso_data_chem$V12),"SNOMEDCT"),]
mrconso_snomed_diso<-mrconso_data_diso[startsWith(as.character(mrconso_data_diso$V12),"SNOMEDCT"),]
mrconso_snomed_proc<-mrconso_data_proc[startsWith(as.character(mrconso_data_proc$V12),"SNOMEDCT"),]
mrconso_snomed_anat<-mrconso_data_anat[startsWith(as.character(mrconso_data_anat$V12),"SNOMEDCT"),]


#function for relation prunning
process1<-function(x,AUIs=AUIs,SCUIs=SCUIs,STR){
	y<-c()
	y[1]<-x[10]
	index1<-fmatch(x[2],AUIs)
	index2<-fmatch(x[6],AUIs)
	y[2]<-as.character(SCUIs[index1])
	y[3]<-as.character(STR[index1])
	y[4]<-as.character(SCUIs[index2])
	y[5]<-as.character(STR[index2])
	y[6]<-x[8]
	y<-unlist(y)
	
}

columns_names<-c("CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF","Last")

colnames(mrconso_data_snomed)<-columns_names

AUIs<-as.character(mrconso_data_snomed$AUI)
SCUIs<-as.character(mrconso_data_snomed$SCUI)
STR<-as.character(mrconso_data_snomed$STR)




#Read MRREL.RFF file
#Please select the path to MRREL file
data<-fread(file="",header=FALSE,sep="|")

AUI1s<-as.character(data$AUI1)
AUI2s<-as.character(data$AUI2)
REL_id<-as.character(data$SRUI)
REL_name<-as.character(data$RELA)

#Extract the realtions between the selected concepts
A<-apply(data1,1,FUN=process1,AUIs,SCUIs,STR)


