
import sys
sys.path.append("..")
import pandas as pd
from ICDCodesGrouper import ICDCodesGrouper
from base import Dataset

class eICU(Dataset):
    
     
    def __init__(self,folder_path=None):
        self.folder_path = folder_path
        self.cache = {}
        self.grouper = ICDCodesGrouper(ccs_path='CCS-SingleDiagnosisGrouper.txt')
        

    def __convert__(self,gem,code):
    
        if code in self.cache.keys():
            return self.cache[code]
        try:
            group = self.grouper.lookup('ccs',code)
            self.cache[code]=code
            return code
        except:
            lookup = gem[gem.icd10cm.str.match(f'{code}.*')]

            if lookup.empty:
                raise Exception(f'No conversion available for {code}')
            self.cache[code]= lookup.sample(1)['icd9cm'].values[0]
            return self.cache[code]
        
    def get_file(self,file_name=None):
        file = lambda x: f'{self.folder_path}/{x}.csv.gz'
        return pd.read_csv(file(file_name))
    
    def get_tabular(self):
        if hasattr(self,'tabular'):
            return self.tabular
        try:
            self.tabular = pd.read_csv('eicu_tabular.csv')
            return self.tabular
        except FileNotFoundError:
            print('eicu_tabular.csv not found. Converting ICD10 to ICD9.')
            
        gem  =pd.read_csv('icd10cmtoicd9gem.csv')
        dataset = lambda x: f'{self.folder_path}/{x}.csv.gz'

        diagnosis = pd.read_csv(dataset('diagnosis'))
        patient = pd.read_csv(dataset('patient'))[['patientunitstayid','patienthealthsystemstayid','uniquepid']]
       
        df= diagnosis.set_index('patientunitstayid')["icd9code"].dropna().apply(lambda x: x.split(',')).explode().apply(lambda x:x.strip()).to_frame('diagnoses')
        df = patient.merge(df,right_index=True,left_on='patientunitstayid')
       
        df.diagnoses = df.diagnoses.apply(lambda x:x.replace('.',''))
        df.diagnoses = df.diagnoses.apply(lambda x: self.__convert__(gem,x))
        df = df.where(lambda x: x["diagnoses"] !='NoDx',axis=0).dropna()
        self.tabular = df
        df.to_csv('eicu_tabular.csv')
        return self.tabular
    
    def get_diagnoses(self,level = 'Visit', grouper = lambda x: x):
        assert level =='Visit' or level =='Patient'
    
        df = self.get_tabular().copy()
        
        df.diagnoses = df.diagnoses.astype(str).apply(grouper) 
        if level == 'Visit':
            df = df.groupby(['uniquepid','patienthealthsystemstayid'])['diagnoses'].apply(list)
            visit_level = df.groupby('uniquepid').apply(list)
            visit_level = visit_level[visit_level.apply(len) >1]  #Exlude those with less than 2 visits
            return visit_level.to_frame('diagnoses')
        #patient_level =  df.groupby(["SUBJECT_ID"])["ICD9_CODE"].apply(list) no patient level for now
        return patient_level
    #Create 'dataset' superclass
    def to_self_supervised(self,train_grouper = lambda x:x,target_grouper:dict = {'phenotype':lambda x:x},replicate_target=False,task:list = ['phenotype'],modalities=['diagnoses']):
        df = self.get_diagnoses(grouper=train_grouper)
         
        if replicate_target:
            df["phenotype"] = df["diagnoses"].apply(lambda x:x[1:]).apply(lambda visit: [target_grouper['phenotype'](x) for x in visit])
        else:
            df["phenotype"] = df["diagnoses"].apply(lambda x:x[-1]).apply(lambda x: target_grouper['phenotype'](x))
            
        df["diagnoses"] = df["diagnoses"].apply(lambda x: x[:-1])

            
        return df
    
    #TODO: mortality at 30 and 90 days
    def get_label(self,target='mortality',base = None):
        valid_targets = ['mortality']
        assert target in valid_targets, "Unknown target"
        patient = self.get_file('patient')
        mortality = patient.set_index('uniquepid').unitdischargestatus.apply(lambda x: 1 if x=='Expired' else 0)
        mortality = mortality.rename('mortality')
        if base is None:
            base = self.get_diagnoses().copy()
        
        df = base
        
        df = df.merge(mortality,right_index=True,left_index=True)
       
        return df