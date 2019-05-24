#!/bin/bash
module load python/anaconda3.5

datadir=/projects/deepcare/deepcare/snomed/umls_extracted_snomed_graphs/Pruned_UMLS_SNOMED/all_embeddings/
#Poincare
python classifier.multi-class.py $datadir/poincare/SNOMEDCT_isa.txt.emb_dims_20.nthreads_1.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/poincare/SNOMEDCT_isa.txt.emb_dims_50.nthreads_1.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/poincare/SNOMEDCT_isa.txt.emb_dims_100.nthreads_1.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/poincare/SNOMEDCT_isa.txt.emb_dims_200.nthreads_1.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/poincare/SNOMEDCT_isa.txt.emb_dims_500.nthreads_1.txt $datadir/scui_tui.tsv 0.25

#metapath
python classifier.multi-class.py $datadir/metapath2vec/snomed.disorder_drugs.emb.e20.out.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/metapath2vec/snomed.disorder_drugs.emb.e50.out.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/metapath2vec/snomed.disorder_drugs.emb.e100.out.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/metapath2vec/snomed.disorder_drugs.emb.e200.out.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/metapath2vec/snomed.disorder_drugs.emb.e500.out.txt $datadir/scui_tui.tsv 0.25

#node2vec
python classifier.multi-class.py $datadir/node2vec/snomed.emb.p1.q1.w20.l40.e20 $datadir/scui_tui.tsv 0.25 $datadir/node2vec/vertex_dict.tsv
python classifier.multi-class.py $datadir/node2vec/snomed.emb.p1.q1.w20.l40.e50 $datadir/scui_tui.tsv 0.25 $datadir/node2vec/vertex_dict.tsv
python classifier.multi-class.py $datadir/node2vec/snomed.emb.p1.q1.w20.l40.e100 $datadir/scui_tui.tsv 0.25 $datadir/node2vec/vertex_dict.tsv
python classifier.multi-class.py $datadir/node2vec/snomed.emb.p1.q1.w20.l40.e200 $datadir/scui_tui.tsv 0.25 $datadir/node2vec/vertex_dict.tsv
python classifier.multi-class.py $datadir/node2vec/snomed.emb.p1.q1.w20.l40.e500 $datadir/scui_tui.tsv 0.25 $datadir/node2vec/vertex_dict.tsv

#Run cui2vec in original and SCUI format
python classifier.multi-class.py $datadir/cui2vec/cui2vec_pretrained.csv.scui_format.txt $datadir/scui_tui.tsv 0.25
python classifier.multi-class.py $datadir/cui2vec/cui2vec_pretrained.csv.node2vec_format.txt $datadir/cui_tui.tsv 0.25

#Med2vec
python classifier.multi-class.py $datadir/med2vec/codeEmb.npy.scui_format.txt $datadir/scui_tui.tsv 0.25 
