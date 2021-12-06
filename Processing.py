
import torch
import re





def parse(G,row,suffix='',connected=True):
    split = row[0].split()
    
    number = split[0]
    idx = {}
    hierarchy = {}
    G.add_edge(f'{suffix}{number}',f'{suffix}{number}')
    codes = row[1].split()
    codes = [f'{suffix}{code}' for code in codes]
    for code in codes:
        hierarchy[code] = number
        if code not in idx.keys():
            idx[code] = len(G.nodes)
        else:
            print("This should never happen..",code)
        G.add_node(code)    

    edges = [(suffix+number,code) for code in codes]
    G.add_edges_from(edges)
    edges = [(code,code) for code in codes]
    G.add_edges_from(edges)
    if connected:
        edges = [(f'{suffix}0',code) for code in codes]
        G.add_edges_from(edges)

    if '.' in number:
        previouses = number.split('.')
        for i in range(len(previouses)-2,-1,-1):
            ancestor = suffix+'.'.join(previouses[:i+1])
            G.add_edge(ancestor,f'{suffix}{number}')
            #if not fully connected. Stop.
            if not connected:
                break
            edges = [(ancestor,code) for code in codes]
            G.add_edges_from(edges)
         

    return number,idx,hierarchy

def get_or_create_diagnosis_ccs_graph(G,f_name,suffix=False,connected=True,return_idx = False):
    file = open(f_name,"r")
    content = file.read()
    file.close()
    groups = re.findall('^([\d\.?]+\s+[A-Za-z ]+.*[-\d]+)\s*?([\dA-Z\s]+)\n',content,flags = 8)
    if suffix:
        suffix = 'D'
    else:
        suffix = ''
    idx = {}
    hierarchy = {}
    for group in groups:
        number,codes,mapping = parse(G,group,suffix=suffix,connected=connected)
        idx = {**codes,**idx}
        hierarchy = {**hierarchy,**mapping}
    for root in range(0,18+1):
        G.add_edge(f'{suffix}0',f'{suffix}{root}')
        
    if return_idx:
        return G,idx,hierarchy
    return G


def get_or_create_procedures_ccs_graph(G,f_name,suffix=False,connected=True):
    file = open(f_name,"r")
    content = file.read()
    file.close()
    groups = re.findall('^([\d\.?]+\s+).*?([\d+\s]+)\n',content,flags = 8)
    if suffix:
        suffix = 'P'
    else:
        suffix = ''
    for group in groups:
        parse(G,group,suffix=suffix,connected=connected)
    for root in range(1,16+1):
        G.add_edge(f'{suffix}0',f'{suffix}{root}')
    return G


    
    
def init_embeddings(embedding_size,poincare = True,G=None,modalities=['diagnoses'],base_path="./embeddings"):
    
  
    if poincare:
        #https://radimrehurek.com/gensim/models/poincare.html
        from gensim.models.poincare import PoincareModel
        x_ontology = torch.empty((0,embedding_size))
        if 'diagnoses' in modalities:
            model = PoincareModel.load(f'{base_path}/diagnosis-poincare-{embedding_size}')
            x_ontology = torch.cat((x_ontology,torch.tensor(model.kv.get_normed_vectors(),dtype=torch.float32)))

        if 'procedures' in modalities:
            model = PoincareModel.load(f'{base_path}procedures-poincare-{embedding_size}')
            x_ontology = torch.cat((x_ontology,torch.tensor(model.kv.get_normed_vectors(),dtype=torch.float32)))
            
    else:
        x_ontology = torch.rand((len(G.nodes),embedding_size))
        
   
    if x_ontology.shape[0] == 0:
        return None
    return x_ontology