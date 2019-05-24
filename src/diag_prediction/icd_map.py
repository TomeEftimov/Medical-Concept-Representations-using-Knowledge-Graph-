import pandas as pd
import os, sys
import pickle as pk

class ICD2Snomed:
    def __init__(self):
        self.icd9_to_snomed = dict()
        #path = 'data/icd9_maps/ICD9CM_SNOMED_MAP_1TO1_201712.txt'
        path = 'data/icd9_maps/ICD9CM_SNOMED_MAP_ALL_201712.txt'
        print('Reading icd9-snomed map from %s ...' % path)
        df = pd.read_csv(path, sep='\t', usecols=['ICD_CODE', 'SNOMED_CID'],
                         dtype = {'ICD_CODE': 'str', 'SNOMED_CID' : 'str'}) 
        for idx, row in df.iterrows():
            icd9_code = row['ICD_CODE'].replace('.', '')
            #print('[%s] -> [%s]' % (row['ICD_CODE'], row['SNOMED_CID']))
            #print('[%s] -> [%s]' % (icd9_code, row['SNOMED_CID']))
            #sys.exit(1)
            self.icd9_to_snomed[icd9_code] = str(row['SNOMED_CID'])
        print('Finished loading.')
        return

    def dump(self, outpath):
        f = open(outpath, "wb")
        pk.dump(self.icd9_to_snomed, f)
        f.close()
        # NOte string dump shows an extra ".)" at end of snomed ids, disabling
        # it 
        f = open(outpath +".tsv", "w")
        for k,v in self.icd9_to_snomed.items():
            f.write(k + "\t" + v + "\n")
        f.close()

        return


    def __getitem__(self, key):
        #print('Returning val %s' % self.icd9_to_snomed[key])
        return self.icd9_to_snomed[key]

def test_icd2snomed():
    icd2snomed = ICD2Snomed()
    assert(icd2snomed['42731'] == '49436004')
    print('PASSED TEST.')
    return

def test_coverage(outpath):
    icd2snomed = ICD2Snomed()
    icd2snomed.dump(outpath)
    path = 'data/mimic3/DIAGNOSES_ICD.csv'
    df = pd.read_csv(path, usecols=['ICD9_CODE'])
    print('Testing ICD9->SNOMED mapping coverage for %s' % path)
    hits = 0
    misses = 0
    f_miss_log = open('tmp.csv', 'w')
    for idx, row in df.iterrows():
        try:
            snomed_cid = icd2snomed[row['ICD9_CODE']]
            hits += 1
        except KeyError:
            misses += 1
            f_miss_log.write('%s\n' % row['ICD9_CODE'])
    f_miss_log.close()
    print('%d hits and %d misses in ICD9->SNOMED mappings ...' % (hits, misses))
    os.system('cat tmp.csv | sort > missing_codes.csv; rm tmp.csv')
    return

if __name__ == '__main__':
    #test_icd2snomed()
    if(len(sys.argv) < 2) :
        print("Usage <outpath>")
    else:
        test_coverage(sys.argv[1])
