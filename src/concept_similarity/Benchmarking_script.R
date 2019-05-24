library(data.table)
library(fastmatch)
library(tcR)
library(lsa)


#Function to calculate cosine similairty for each pair in the benchamrk files
bootstraping_cosine_similarity<-function(SCUI,SCUI1,SCUI2,data_table){
similarity<-c()
for(i in 1:length(SCUI1))
{
	
	index1<-fmatch(SCUI1[i],SCUI)
	index2<-fmatch(SCUI2[i],SCUI)
	
	if(!is.na(index1) && !is.na(index2)){
		similarity[i]<-cosine(as.numeric(data_table[index1,]),as.numeric(data_table[index2,]))
	}

	if(is.na(index1) || is.na(index2)){
		similarity[i]<-100
	}
	
}

return(similarity)


}


#Select one of the benchmark files: D1 (CHEM,CHEM) - isA.txt, D2 (DISO,DISO) - isA.txt, D3 (CHEM,CHEM)- all.txt, D4 (DISO, DISO)- all.txt, D5 (DISO, CHEM).txt

data<-fread(file="/Benchmark files/D1 (CHEM,CHEM) - isA.txt",header=TRUE,sep="|")
SCUI1<-as.character(data$SCUI1)
SCUI2<-as.character(data$SCUI2)


#Metapath2vec

#Select the embeddings file: it can be 20, 50, 100, 2000, and 500 dimenisonal embeddings

metapath2vec<-fread(file="/Embeddings/metapath2vec/snomed.disorder_drugs.emb.e100.out.txt",sep=" ",header=FALSE)
SCUI<-as.character(metapath2vec$V1)
SCUI<-SCUI[2:length(SCUI)]

for(i in 1:length(SCUI)){
	SCUI[i]<-substr(SCUI[i],2,nchar(SCUI[i]))
}

data_temp<-metapath2vec[2:nrow(metapath2vec),2:ncol(metapath2vec)]
data_table<-data_temp

meta<-bootstraping_cosine_similarity(SCUI,SCUI1,SCUI2,data_table)





#Node2vec

#Select the embeddings file: it can be 20, 50, 100, 2000, and 500 dimenisonal embeddings


node2vec<-fread(file="/Embeddings/node2vec/snomed.emb.p1.q1.w20.l40.e100",sep=" ",header=FALSE)

#Select the vertex dictionary to map the embedding index to SNOMED CT unique identifier

vertex_scui<-fread(file="/Embeddings/vertex_dict.tsv",sep="	",header=FALSE)
ids<-as.character(vertex_scui$V2)
scui<-as.character(vertex_scui$V1)

data_temp<-node2vec[,2:ncol(node2vec)]
id<-as.character(node2vec$V1)




SCUI<-c()
for(i in 1:length(id)){
	
	index<-fmatch(id[i],ids)
	SCUI[i]<-scui[index]
}

data_table<-data_temp

node<-bootstraping_cosine_similarity(SCUI,SCUI1,SCUI2,data_table)




#Poincaree 

#Select the embeddings file: it can be 20, 50, 100, 2000, and 500 dimenisonal embeddings
poincare<-fread(file="/Embeddings/poincare/SNOMEDCT_isa.txt.emb_dims_100.nthreads_1.txt",sep=" ",header=FALSE)
SCUI<-as.character(poincare$V1)
data_temp<-poincare[,2:ncol(poincare)]
data_table<-data_temp

poincare<-bootstraping_cosine_similarity(SCUI,SCUI1,SCUI2,data_table)


#Caluclate the statistical power

metapath2vec_power<-sort(meta)[0.05*length(meta)]
node2vec_power<-sort(node)[0.05*length(node)]
poincare_power<-sort(poincare)[0.05*length(poincare)]

#Plot the bootstrap distributions of the cosine similarity


plot(density(meta),ylim=c(0,14),main="",lty=1,lwd=3,col="red")
lines(density(node),main="",lty=1,lwd=3,col="blue")
lines(density(poincare),main="",lty=1,lwd=3,col="green")
legend("topleft",legend=c("Metapath2vec", "Node2vec","Poincare"),col=c("red", "blue","green"), lty=c(1,1,1),lwd=c(2.5,2.5,2.5), cex=1)


#In the case of cui2vec embeddings, since they are only 500D, the subsets of the benchamrking files and the comparison are avaible with the following code


#Cui2vec

#Select the cui2vec embeddings files

cui2vec<-fread(file="/Embeddings/cui2vec/cui2vec_pretrained.csv",sep=",",header=FALSE)
cui2vec<-cui2vec[2:nrow(cui2vec),]
cuis<-as.character(cui2vec$V1)

#Select cui to scui mapping dicitonary
cui_scui<-fread(file="/Embeddings/cui2vec_to_scui2vec.txt",header=TRUE,sep="|")
SCUI<-as.character(cui_scui$cuis_to_scuis)


data_table<-cui2vec[,2:ncol(cui2vec)]
data_table<-data.matrix(data_table)


cui2vec_sim<-bootstraping_cosine_similarity(SCUI,SCUI1,SCUI2,data_table)


indices<-which(cui2vec_sim!=100)

node<-node[indices]
meta<-meta[indices]
poincare<-poincare[indices]
cui2vec_sim<-cui2vec_sim[indices]


metapath2vec_power<-sort(meta)[0.05*length(meta)]
node2vec_power<-sort(node)[0.05*length(node)]
poincare_power<-sort(poincare)[0.05*length(poincare)]
cui2vec_power<-<-sort(cui2vec_sim)[0.05*length(cui2vec_sim)]




