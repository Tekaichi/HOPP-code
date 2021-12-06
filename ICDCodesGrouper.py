import pandas as pd
import numpy as np
import re
from abc import ABC,abstractmethod 
import pickle
from bs4 import BeautifulSoup
import requests


class ICDLookup(ABC):
    
    @abstractmethod 
    def lookup(self,code):
        pass
    
    @property
    def classes(self):
        pass  
    
class ICDCodesGrouper(object):
    """
    Class containing several icd9 grouping subclasses.
    all grouper subclasses implement the method "lookup".
    
    <lookup> can accept as input:
    1- a single icd9 code as string
    2- a pd.Series of icd9 codes
    
    and outputs a mapping to the corresponding icd9 group.
    
    
    Examples
    --------
    
    >>> codes = pd.Series([5849,E8497,2720])
    
    >>> grouper = ICDCodesGrouper(ccs_path,icd9_chapter_path)
    >>> grouper.check_avaliable_groupers()
    ['ccs', 'icd9chapters']
    
    >>> grouper.lookup('ccs',codes)
    0     157
    1    2621
    2      53
    dtype: int64
    
    >>> grouper.lookup('icd9chapters',codes)
    0    10
    1    19
    2     3
    dtype: int64
    """
    
    def __init__(self,ccs_path = None,icd9_chapter_path = None,category_level=True,ccs_procedure_path = None):
        """
        Parameters
        ----------
        data paths to each grouping method
        """
        self.groupers ={}
        if ccs_path:
            self.ccs = self.CCSSingleDiagnosis(ccs_path)
            self.groupers['ccs'] = self.ccs
        if ccs_procedure_path:
            self.ccs_proc = self.CCSSingleDiagnosis(ccs_procedure_path)
            self.groupers['ccs_procedure'] = self.ccs_proc
        if icd9_chapter_path:
            self.icd9chapters = self.ICD9_CM_Chapters(icd9_chapter_path)
            self.groupers['icd9chapters'] = self.ccs
        if category_level:
            self.category_level = self.SecondHierarchyICD9()
            self.groupers['cat_level']= self.category_level
            
        self.icd9_level3 = self.ICD9_LEVEL3()
        
        self.groupers['icd9_level3'] =   self.icd9_level3
        self.raw = self.Raw()
        self.groupers['all'] = self.raw
    @property
    def get_available_groupers(self):
        #return self.groupers.keys?
        return [i for i in self.groupers]
    
    def get_classes(self,grouper:str):
        return self.groupers[grouper].classes
    
    def lookup(self,grouper: str,code):
        """
        
        Parameters
        ----------
        grouper : str
            grouper must exist in self.check_avaliable_groupers
        code : str | pd.Series
            icd9 code or pd.Series of codes
        """

        if grouper not in self.groupers.keys():
            raise ValueError(f'Expecting one of the following \
                            groupers: {self.get_available_groupers},\
                            got instead {grouper}')
        
        try:
            return self.groupers[grouper].lookup(code)
        except Exception as msg:
            print(f'{code} for {grouper} failed with {msg}')
    
    class ICD9_LEVEL3:
        """
        maps icd9 codes to the first 3 levels
        """
        
        def __init(self):
            pass
        @property
        def classes(self):
            with open("icd3_level.pkl", "rb") as fp:   #Pickling
                return pickle.load(fp)
        
        def lookup(self,code):
            def digit_3(code):
                #return code[:3]
                if code[0] in ['V','E']:
                    return code[:4]
                return code[:3]
            #E or V as prefix?
            if type(code) == list:
                code = pd.Series(code)
                return code.astype(str).apply(digit_3).values
            elif type(code) == pd.Series:
                code_level3 = code.astype(str).apply(digit_3)
                assert code_level3.apply(len).unique()[0] == 3,f'Oops. Got {code_level3.apply(len).unique()}'
            else:
                code_level3 = digit_3(str(code))
                assert len(code_level3) == 3,f'Oops. Got {code_level3}'
            return code_level3
    #TODO            
    class Raw(ICDLookup):
        def __init__(self):
            #todo
            #get all classes (read from that .txt file)
            with open('icd9.pkl','rb') as fp:
                icd9 = pickle.load(fp)
                self.all_icd9 = [item[0] for item in icd9]
                self.names = dict((x,y) for x,y in icd9)
        
        @property
        def classes(self):
            return self.all_icd9
        
        def lookup(self,code):
            return code
        
    class SecondHierarchyICD9(ICDLookup):
        #TODO: Save scrapped file.
        
        """
        Maps icd9 codes to 2nd hierarchy icd9
        """
        def __init__(self):
            base_url = 'http://www.icd9data.com/2015/Volume1/default.htm'
            page = requests.get(base_url)
            soup = BeautifulSoup(page.content, "html.parser")
            definition = soup.find('div',class_='definitionList')

            chapters = definition.find_all('a',class_='identifier')
            dic = {}
            for chapter in chapters:
                page = requests.get(f'http://www.icd9data.com{chapter["href"]}')
                soup = BeautifulSoup(page.content, "html.parser")
                definition = soup.find(class_='definitionList')
                res = definition.find_all('a',class_='identifier')
                dic[chapter.text.strip()] = [item.text.strip() for item in res]

            self.second_hierarchy = sum(dic.values(),[])
            self.dic = dic
            bins = pd.Series(self.second_hierarchy).apply(lambda x: x.split('-')).apply(max).sort_values()
            bins = bins.to_frame('bins')
            bins['label'] = pd.Series(self.second_hierarchy)
            self.v_bins = bins.copy()[bins['bins'].apply(lambda x: x[0])=='V']
            self.e_bins = bins.copy()[bins['bins'].apply(lambda x: x[0])=='E']
            self.bins = bins.copy()[bins['bins'].apply(lambda x: x[0].isdigit())]
            
            #TODO:
            #Save and load bins.
            self.v_bins['bins'] =  self.v_bins['bins'].apply(lambda x:x[1:]).astype(int)
                       
            self.e_bins['bins'] =  self.e_bins['bins'].apply(lambda x:x[1:]).astype(int)
 
            self.bins['bins'] =   self.bins['bins'].astype(int)
      
            self.cache_lookup = {}
        @property
        def classes(self):
            return np.array(self.second_hierarchy)
        
        def lookup(self,code):
            def single_lookup(code_):
                
                #cache system
                if code_ in self.cache_lookup.keys():
                    return self.cache_lookup[code_]
                code_ = code_[:3]
                def char_lookup(code_):
                    """
                    When the code starts by a char, it's either E or V
                    """
                    rel_code = int(code_[1:])
                    if code_[0] == 'E':
                        pos =  np.digitize(rel_code,self.e_bins["bins"],right=True)
                        return self.e_bins.iloc[pos].label
                    elif code_[0] == 'V':
                        pos =  np.digitize(rel_code,self.v_bins["bins"],right=True)
                        return self.e_bins.iloc[pos].label
                    
                    raise ValueError(f'{code_} was not recognized')

                def int_lookup(code_):
                    level_3_code = int(code_[:3])
                    pos = np.digitize(level_3_code,self.bins["bins"],right=True)
                    return self.bins.iloc[pos].label
                
                if code_[0] in ['E','V']:
                    result= char_lookup(code_)
                    self.cache_lookup[code_] =result
                    return result

                try:
                    int(code_)
                except:
                    raise ValueError(f'Expecting a code starting with "E", "V", or an integer compatible string. Got instead {code_}')
                    
                result = int_lookup(code_)
                
                self.cache_lookup[code_] =result
                return result
            #buggy?
            def batch_lookup(code_:pd.Series):
                #Separate digits from non digits and run digitize max 3 times instead of |N|
                return code_.apply(single_lookup)
            
            if type(code) == list:
                code = pd.Series(code)
                return code.apply(single_lookup).values
            if type(code) == pd.Series:
                return bach_lookup(code)
            elif type(code) == str:
                return single_lookup(code)
     
 
        
    class CCSSingleDiagnosis(ICDLookup):
        """
        Maps icd9 codes to CCS groups
        """
        def __init__(self,file = None):
            #Remove this line
            assert file is not None, 'No file to read from'
            file = open(file,"r")
            content = file.read()
            file.close()
            lookup = {}
            groups = re.findall('^(\d{1,4}?)[ ]+?(.*)\s+([\s\dA-Z \r\n]+)$',content,re.MULTILINE)
            self.names = {}
            for group in groups:           
                codes = group[2].split()
                name = group[1].strip()
                group = int(group[0]) #group code
                for code in codes:
                    lookup[code] = group
                self.names[group] = name
            self._lookup_table = lookup
            
        @property
        def classes(self):
            classes = np.array(list(set(self._lookup_table.values())))
            classes.sort()
            return classes
        def lookup(self,code):
            """
            Given an icd9 code, returns the corresponding ccs code.
            
            Parameters
            ----------
            
            code : str | pd.Series
                icd9 code
            
            Returns:
                -1: code received isn't a string
                0: code doesn't exist
                >0: corresponding ccs code
            """
            
            def lookup_single(code : str):
                try:
                    if type(code) == str:
                        return self._lookup_table[code]
                    else:
                        raise ValueError(f'{code} is not a string')
                except KeyError:
                    #Append 0 and try again?
                    raise ValueError(f'{code} not found. Maybe its not in ICD-9 format?')
              
                    
            if type(code) == list:
                code = pd.Series(code)
                return code.apply(lookup_single).values
            if type(code) == pd.Series:
                return code.apply(lookup_single)
            elif type(code) == str:
                return lookup_single(code)
            else:
                raise ValueError(f'Wrong input type. Expecting str or pd.Series. Got {type(code)}')
        
    class ICD9_CM_Chapters:
        """
        Maps icd9 codes to icd9 chapters
        """
        def __init__(self,filepath):
            # creates self.chapters_num & self.chapters_char & self.bins
            self.__preprocess_chapters(filepath)

        def lookup(self,code):
            """
            
            
            Parameters
            ----------
            
            code : str | pd.Series

            Returns
            -------
            
            chapter : str | pd.Series
                Corresponding icd9 chapter
            """

            def single_lookup(code_):

                def char_lookup(code_):
                    """
                    When the code starts by a char, it's either E or V
                    """
                    if code_[0] == 'E':
                        return 19
                    elif code_[0] == 'V':
                        return 18
                    return 0

                def int_lookup(code_):
                    level_3_code = int(code_[:3])
                    pos = np.digitize(level_3_code,self.bins)
                    chapter = self.chapters_num.Chapter.iloc[pos-1]
                    return chapter

                if code_[0] in ['E','V']:
                    return char_lookup(code_)
                
                try:
                    #code_ = 
                    int(code_)
                except:
                    raise ValueError(f'Expecting a code starting with "E", "V", or an integer compatible string. Got instead {code_}')
                
                return int_lookup(code_)

            def batch_lookup(codes : pd.Series):
                
                # to sort everything at the end
                original_order = codes.index.copy()
                
                mask_is_alpha = codes.apply(lambda x: (x[0] == 'E') | (x[0] == 'V') if not pd.isna(x) else False)
                codes_char = (codes
                              .loc[mask_is_alpha]
                              .copy()
                              .apply(lambda x:x[0]) # only need first character to identify chapter
                             )
                
                codes_nan = codes[pd.isna(codes)]
                codes_num = (codes
                             .loc[~codes.index.isin(codes_char.index.tolist() + codes_nan.index.tolist())]
                             .copy()
                             .apply(lambda x: x[:3]) # only need first 3 characters to identify chapter
                             .astype(int)
                            )
                
                
                # get chapters of numeric codes
                num_chapters = (pd.Series(data=np.digitize(codes_num,self.bins),
                                          index=codes_num.index)
                               )
                
                
                char_chapters = codes_char.apply(single_lookup)
                result = (pd.concat([num_chapters,char_chapters,codes_nan],axis='rows') # merge chapters of numerical & alpha codes
                          .loc[original_order] # get original order
                         )
                return result
            
            if type(code) not in [str,pd.Series]:
                return -1

            if type(code) == str:
                return single_lookup(code)
            elif type(code) == pd.Series:
                return batch_lookup(code)
            else:
                raise ValueError(f'Expecting code to be either str or pd.Series. Got {type(code)}')

        def __preprocess_chapters(self,filepath):
            """
            Some preprocessing to optimize assignment speed later using np.digitize
            1. Separate chapters into numeric codes or alpha codes (There are two chapters considered alpha because they start with "E" or "V")
            2. Create self.bins, which contains starting code ranges of each chapter
            """
            
            self.chapters = pd.read_csv(filepath)

            # preprocess chapters dataframe: split into alpha vs numeric
            self.chapters['start_range'] = self.chapters['Code Range'].apply(lambda x: x[:x.find('-')])

            chapters_char = (self.chapters
                           .loc[self.chapters.start_range
                                .apply(lambda x: len(re.findall('^(E|V)',x)) > 0),
                                ['Chapter','Description','Code Range']
                               ]
                          )
            # only need the first letter
            chapters_char.loc[:,'Code Range'] = chapters_char['Code Range'].apply(lambda x: x[0])

            chapters_num = (self.chapters
                          .loc[~self.chapters.index.isin(chapters_char.index)]
                          .astype({'start_range':int}).copy()
                         )

            bins = chapters_num.start_range.tolist()
            # need to add last interval
            bins.append(bins[-1]+205)

            self.bins = bins
            self.chapters_num = chapters_num
            self.chapters_char = chapters_char