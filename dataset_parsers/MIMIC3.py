import pandas as pd
from .base import Dataset

class MIMIC_3(Dataset):
   
    def __init__(self,folder_path=None):
        self.folder_path = folder_path
 
    def get_file(self,file_name=None):
        file = lambda x: f'{self.folder_path}/{x}.csv.gz'
        return pd.read_csv(file(file_name),index_col=0,dtype=object)
    
    def get_tabular(self,modalities = ['diagnoses'],dropna=True):
   
        assert len(modalities) >0, "No modalities chosen"
        admissions = self.get_file('ADMISSIONS')[["SUBJECT_ID","HADM_ID"]]
        
        if 'diagnoses' in modalities:
            diagnoses = self.get_file('DIAGNOSES_ICD')[['HADM_ID','SUBJECT_ID','ICD9_CODE']]
            admissions = admissions.merge(diagnoses,right_on=["SUBJECT_ID","HADM_ID"],left_on=["SUBJECT_ID","HADM_ID"]).rename(columns={'ICD9_CODE':'diagnoses'})
           
        if 'procedures' in modalities:
            procedures = self.get_file('PROCEDURES_ICD')[['HADM_ID','SUBJECT_ID','ICD9_CODE']]
            admissions = admissions.merge(procedures,right_on=["SUBJECT_ID","HADM_ID"],left_on=["SUBJECT_ID","HADM_ID"]).rename(columns={'ICD9_CODE':'procedures'})
           
        #self.tabular = admissions.dropna()
        if dropna:
            return admissions.dropna()
        else:
            return admissions
    def get_diagnoses(self,level = 'Visit', grouper = lambda x: x):
       
        return self.__get_modality__(level=level,grouper=grouper)
    
    def __get_modality__(self,modality='diagnoses',grouper=lambda x:x,level='Visit'):
        assert level =='Visit' or level =='Patient'
    
        df = self.get_tabular(modalities=[modality]).copy()
        
        df[modality] = df[modality].astype(str).apply(grouper) 
        
        if level == 'Visit':
            visit_level = df.groupby([ "SUBJECT_ID","HADM_ID"])[modality].apply(list).groupby('SUBJECT_ID').apply(list)           
            return visit_level.dropna()

        patient_level =  df.groupby(["SUBJECT_ID"])[modality].apply(list)
        return patient_level
    
    def get_procedures(self, level='Visit',grouper=lambda x:x):
        return self.__get_modality__(modality='procedures',level=level,grouper=grouper)
        
    
    def to_self_supervised(self,train_grouper = lambda x:x,target_grouper:dict = {'phenotype':lambda x:x},replicate_target=False,task:list = ['phenotype'],modalities=['diagnoses']):
        assert len(modalities) >0 
        assert 'diagnoses' in modalities or 'procedures' in modalities
                    
        df = self.__get_modality__(modality=modalities[0],grouper=train_grouper).to_frame(modalities[0])
        if len(modalities) ==2:                
            other = self.__get_modality__(modality=modalities[1],grouper=train_grouper).to_frame(modalities[1])
           
            df = df.merge(other,right_index=True,left_index=True,how='outer')
            
            mask = df[modalities[0]].apply(lambda x: 0 if type(x) is float else len(x)) >1
            for modality in modalities[1:]:
                mask = mask | (df[modality].apply(lambda x: 0 if type(x) is float else len(x)) > 1)
            df = df[mask]
        else:
            df = df.dropna()[df[modalities[0]].apply(lambda x: 0 if type(x) is float else len(x)) >1]  #Exlude those with less than 2 visits
                     
        #If modalities == 2, then its diagnoses and procedures. 
        if len(modalities) ==2:
            #make it such that |diagnoses| == |procedures|
            def pad(sml,bg,x):
                try:
                    nw = x[sml].copy()
                except AttributeError:
                    nw = [[]] #Initialize with one empty visit.
                try:
                    bg = len(x[bg])
                except TypeError:
                    bg = 1
                try:    
                    sml = len(x[sml])
                except TypeError:
                    sml = 1
                    
                delta = bg -sml
                if delta <= 0:
                    return nw
                nw = nw + [[] for _ in range(delta)]
                return nw
            
            procedures = df.apply(lambda x: pad('procedures','diagnoses',x),axis=1)
            diagnoses = df.apply(lambda x: pad('diagnoses','procedures',x),axis=1)
            df.procedures= procedures
            df.diagnoses = diagnoses
        
        
        if replicate_target:
            if 'phenotype' in task:
                df["phenotype"] = df["diagnoses"].apply(lambda x:x[1:]).apply(lambda visit: [target_grouper['phenotype'](x) for x in visit])
            if 'treatment' in task:
                df["treatment"] = df["procedures"].apply(lambda x:x[1:]).apply(lambda visit: [target_grouper['treatment'](x) for x in visit])
        else:
            if 'phenotype' in task:
                df["phenotype"] = df["diagnoses"].apply(lambda x:x[-1]).apply(lambda x: target_grouper['phenotype'](x))
            if 'treatment' in task:
                df["treatment"] = df["procedures"].apply(lambda x:x[-1]).apply(lambda x: target_grouper['treatment'](x))
            
        for modality in modalities:
            df[modality] = df[modality].apply(lambda x: x[:-1])
            
        return df.dropna()
    
    def get_label(self,target:str='mortality',base:pd.DataFrame = None):
        
        assert target in ['mortality','mortality@30','mortality@90'], "Expecting mortality tasks"

        if base is None:
            base = self.get_diagnoses().copy().to_frame('diagnoses')
        
        df = base
        if target == 'mortality':
            admissions = self.get_file('ADMISSIONS')[["SUBJECT_ID","DEATHTIME"]]
            admissions["DEATHTIME"] = admissions["DEATHTIME"].notnull().astype('int')
            admissions = admissions.groupby('SUBJECT_ID')["DEATHTIME"].apply(max)
            admissions = admissions.rename("mortality")
        else:
            days = int(target.split("@")[1])
            admissions = self.get_file('ADMISSIONS')[["SUBJECT_ID","ADMITTIME","DEATHTIME"]]
          
            admissions.DEATHTIME = pd.to_datetime(admissions.DEATHTIME)
            admissions.ADMITTIME = pd.to_datetime(admissions.ADMITTIME)
            def get_diff(lst):
                if len(lst)==1:
                    return None
                else:
                    return (lst[-1]-lst[-2]).days
            #Is this really working?
            died = admissions[admissions["DEATHTIME"].notnull()]["SUBJECT_ID"]
            mask = ~admissions["DEATHTIME"].notnull() & admissions['SUBJECT_ID'].isin(died)
            admissions.loc[mask,'DEATHTIME'] =admissions[mask].ADMITTIME
            days_between = admissions.groupby('SUBJECT_ID')["DEATHTIME"].apply(list).apply(get_diff).dropna()
            admissions = days_between.apply(lambda x: x>days).astype(int).to_frame(f'mortality@{days}')
                
        df = df.merge(admissions,right_index=True,left_index=True,how='inner')
        return df