#!/bin/bash
#SBATCH -N 1
#SBATCH -t 23:59:00
##SBATCH -A deepcare
##SBATCH -p short
#SBATCH -J node2vec
#SBATCH -o snomed.out.%j
#SBATCH -e snomed.err.%j
#SBATCH -p dl
#SBATCH --gres=gpu:2
##SBATCH -n 4

##NOTE : this script runs on marianas not constance regular nodes
module purge
module load cuda
module load python/anaconda3
#dirpath=/projects/deepcare/deepcare/snomed/snomed_graphs/reduced.v2/intGraph/
#input=$dirpath/snomed_edges.int.txt
dirpath=/projects/deepcare/deepcare/snomed/umls_extracted_snomed_graphs/Pruned_UMLS_SNOMED/int_graph_node2vec/
input=$dirpath/umls_snomed_scui.int.tsv
p=1
q=1
numWalks=20
walkLen=40
emb=500
mkdir $dirpath/emb/
mkdir $dirpath/walks/
embOut=$dirpath/emb/snomed.emb.p$p.q$q.w$numWalks.l$walkLen.e$emb
walkOut=$dirpath/walks/snomed.walks.p$p.q$q.l$walkLen.w$numWalks
python /projects/deepcare/deepcare/node2vec/src/main.py --input $input --output $embOut --walkOutput $walkOut --dimensions $emb  --walk-length $walkLen --num-walks $numWalks

for numWalks in 10 20 40
do
		for walkLen in 20 40 80
		do
				embOut=$dirpath/emb/wordnet.emb.p$p.q$q.w$numWalks.l$walkLen.e$emb
				walkOut=$dirpath/walks/snomed.walks.p$p.q$q.l$walkLen.w$numWalks
				python src/main.py --input $input --output $embOut --walkOutput $walkOut --dimensions $emb  --walk-length $walkLen --num-walks $numWalks
		done
done
#for p in 0.25 0.50 1 2 4
#do
#  for q in 0.25 0.50 1 2 4
#  do
#	embOut=$dirpath/emb/wordnet.emb.p$p.q$q.w$numWalks.l$walkLen.e$emb
#	walkOut=$dirpath/walks/wordnet.walks.p$p.q$q.l$walkLen.w$numWalks
#	python src/main.py --input $input --output $embOut --walkOutput $walkOut --dimensions $emb  --walk-length $walkLen --num-walks $numWalks
#  done
#done
