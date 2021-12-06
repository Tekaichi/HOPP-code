from dataset_parsers.MIMIC3 import MIMIC_3
from dataset_parsers.eICU import eICU
from ICDCodesGrouper import ICDCodesGrouper
import pickle
import wandb
from os import path
import os
import pandas as pd
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='Execute Phenotype prediction model')


def str2bool(value):
    value = value.lower()
    return  value == 'true'
    


parser.add_argument('-l','--layers', dest='n_layers', type=int, nargs='+',
                    help='Number of layers',default=[0,1,2,3])

parser.add_argument('-e','--embeddings', dest='embedding_sizes', type=int, nargs='+',
                    help='Embedding Sizes',default=[50,128,256,512])


parser.add_argument("-b", "--batch",type=int,
                  dest="batch_size",
                  help="Batch size",default=256)


parser.add_argument("--epochs",type=int,
                  dest="epochs",
                  help="Epochs",default=25)



parser.add_argument("--dataset",type=str,
                  dest="dataset",
                  help="Dataset path to use",default='mimic')

parser.add_argument("--replicate",type=str2bool,
                  dest="replicate",
                  help="Use target replication",default="False")


parser.add_argument("--grouper",type=str,
                  dest="grouper",
                  help="Select grouper",default='ccs')


parser.add_argument("--directed",type=str2bool,
                  dest="directed",
                  help="Directed Graph",default="True")


parser.add_argument("--add_labels",type=str2bool,
                  dest="add_labels",
                  help="Add missing labels",default="False")

parser.add_argument("--causal",type=str2bool,
                  dest="causality",
                  help="True if task[i] depends on task[i-1]",default="False")

parser.add_argument("--modalities",type=str,nargs="+",
                  dest="modalities",
                  help="Select modalities to be used",default=['diagnoses'])



parser.add_argument("--masking",type=float,
                  dest="masking",
                  help="% masking",default=0.0)



parser.add_argument('--dropout',type=str2bool,
                   dest='dropout',
                   help="use dropout",default='true'
                   )

parser.add_argument("--task",type=str,nargs='+',
                  dest="task",
                  help="Tasks",default=['phenotype'])

parser.add_argument('--ancestry',type=str,
                   dest='ancestry',
                   help="ancestry type",default="full"
                   )

parser.add_argument("--override",type=bool,
                  dest="override",
                  help="Run again",default=False)

parser.add_argument("--k_fold",type=int,
                  dest="k_fold",
                  help="Select k for k-fold cross validation",default=0)

#one for each task.
parser.add_argument("--optimize",type=str,nargs='+',
                  dest="optimize",
                  help="Metric to be optimized",default=["R@30"])


parser.add_argument('--poincare',type=str2bool,
                   dest='poincare',
                   help="Use poincare",default='true'
                   )

parser.add_argument("-c",type=str2bool,
                  dest="cooccurrence",
                  help="co-ocurrence directed modality",default="false")

args = parser.parse_args()
print('Running with',vars(args))


assert args.masking >= 0 and args.masking  <= 1 , "Masking probability needs to be between [0,1]"
assert args.dataset.lower() in ['mimic','eicu','all'], 'Unknown dataset, please use mimic or eicu'
assert args.k_fold == 0 or (args.k_fold > 0 and args.optimize is not None)
assert 'diagnoses' in args.modalities or 'procedures' in args.modalities, 'No known modalities'
assert ('procedures' in args.modalities and args.dataset == 'mimic') or 'procedures' not in args.modalities, 'eICU does not have procedures'
assert args.k_fold == 0 or (args.k_fold >1 and len(args.optimize) == len(args.task)),"If k-fold >1, optimize metrics length needs to match length of tasks"
assert args.ancestry in ['no','full','partial'],"Ancestry parameter not recognized"
assert 'treatment' not in args.task or ('treatment' in args.task and args.grouper !='cat_level')

icdgrouper = ICDCodesGrouper(ccs_path='./icd_files/CCS-SingleDiagnosisGrouper.txt',ccs_procedure_path='./icd_files/CCS-SingleProceduresGrouper.txt')

if args.dataset.lower() == 'mimic':
    mimic_path =  "/Datasets/MIMIC_3/"
    dataset = MIMIC_3(mimic_path,save_steps = True)
elif args.dataset.lower()=='eicu':
    eicu_path = "/Datasets/eICU/"
    dataset = eICU(eicu_path,save_steps = True)
elif args.dataset.lower()=='all':
    assert 'phenotype' in args.task and len(args.task)==1, "Can't use both datasets for non-phenotype tasks"
    dataset = []
    mimic_path =  "/Datasets/MIMIC_3/"
    dataset.append(MIMIC_3(mimic_path,save_steps = True))
    eicu_path = "/Datasets/eICU/"
    dataset.append(eICU(eicu_path,save_steps = True))

grouper = lambda group: lambda x: icdgrouper.lookup(group,x)

args.modalities.sort() #does this work?

df = None
grouper = {"phenotype":grouper(args.grouper),"treatment":grouper(f'{args.grouper}_procedure')} 
n_labels = []

#move this IF to loop below. Need todo ICD Single Grouper for procedures
from sklearn.preprocessing import MultiLabelBinarizer

if 'phenotype' in args.task or 'treatment' in args.task:
    if args.dataset.lower()=='all':
        df = pd.concat([item.to_self_supervised(target_grouper = grouper,replicate_target=args.replicate,task = args.task,modalities=args.modalities) for item in dataset])
    else:
        df = dataset.to_self_supervised(target_grouper = grouper,replicate_target=args.replicate,task = args.task,modalities=args.modalities)
    
for task in args.task:
    if 'mortality' in task:
        df = dataset.get_label(base=df,target=task)
        n_labels.append(1)#Binary task
    
    if 'phenotype' in task or 'treatment' in task:
        if args.add_labels:
            rel_grouper = f'{args.grouper}_procedure' if 'treatment' in task else args.grouper
            all_classes = icdgrouper.get_classes(rel_grouper)
            mlb = MultiLabelBinarizer(classes=all_classes)
        else:
            mlb = MultiLabelBinarizer()

    
        if args.replicate:
            target = sum(df[task],[])
            szs = df[task].apply(len)
        else:
            target = df[task]
            
        multi_hot = mlb.fit_transform(target)
        
        n_labels.append(multi_hot.shape[1])
        
        
        if args.replicate:
            curr = 0
            result = []
            for idx,sz in enumerate(szs):
                item = multi_hot[curr:curr+sz]
                curr = curr + sz
                assert len(item) >0,f'{idx} is empty'
                result.append(item)
            multi_hot = result
     
        df[task] = list(multi_hot)
    


from torch_geometric.data import Data
import networkx as nx
import re
from metrics import * 
from utilities import * 


from sklearn.model_selection import train_test_split
from Processing import init_embeddings,get_or_create_diagnosis_ccs_graph,get_or_create_procedures_ccs_graph


epochs = args.epochs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using",device)


embedding_sizes = args.embedding_sizes
n_layers = args.n_layers
masking = args.masking
random.shuffle(embedding_sizes)
random.shuffle(n_layers)


dataset = df
train,test = train_test_split(dataset,test_size=0.25)

def to_pair(diagnoses,procedures):
    res = []
    for (item1,item2) in zip(diagnoses,procedures):
        temp = []
        for(item_a,item_b) in zip(item1,item2):
            temp.append([item_a,item_b])
        res.append(temp)
    return np.array(res)

def data_to_input(train,test):

    if len(args.modalities) == 2:
        diagnoses_train = train.diagnoses.values
        procedures_train = train.procedures.values

        diagnoses_validation = test.diagnoses.values
        procedures_validation = train.procedures.values
        x_train = to_pair(diagnoses_train,procedures_train)
        x_validation = to_pair(diagnoses_validation,procedures_validation)

    else:
        x_train = train[args.modalities[0]].values
        x_validation = test[args.modalities[0]].values

    y_train = {}
    y_validation ={}
    for task in args.task:
        dim = len(np.array(list(dataset[task]),dtype=object).shape)
        y_train[task] = np.array(list(train[task]))
        y_validation[task] = np.array(list(test[task]))
        if 'mortality' in task: #atm only binary task is mortality[ |30|90]
            y_train[task] = y_train[task].reshape(-1,1)
            y_validation[task] = y_validation[task].reshape(-1,1)
            
    return x_train,x_validation,y_train,y_validation

batch_size = args.batch_size
replicate = args.replicate


if args.directed:
    G = nx.DiGraph()
else:
    G = nx.Graph()
    
fd_name = "CCS-MultiDiagnosisGrouper.txt"
fp_name = "CCS-MultiProceduresGrouper.txt"


if len(args.modalities) == 2:
    G = get_or_create_diagnosis_ccs_graph(G,fd_name,connected=args.ancestry =='full',suffix='D')
    G = get_or_create_procedures_ccs_graph(G,fp_name,connected=args.ancestry =='full',suffix='P')
    from graph_builders.double import DoubleGraph as Graph
else:
    
    if args.cooccurrence:
        from graph_builders.singleuni import SingleGraph as Graph
    else:
        from graph_builders.single import SingleGraph as Graph
    if args.modalities[0] =='diagnoses':
        if args.ancestry == 'no': #TODO: something like this
            G,idx,hierarchy = get_or_create_diagnosis_ccs_graph(G,fd_name,connected=True,return_idx=True) #use idx in init_embeddings thing..
            G =nx.subgraph(G, list(idx.keys()))
        else:
            G = get_or_create_diagnosis_ccs_graph(G,fd_name,connected=args.ancestry =='full')

    else:
        G = get_or_create_procedures_ccs_graph(G,fp_name,connected=args.ancestry)


from models import OutcomeModel as Net

aux = 'replicate' if args.replicate else 'normal'


dropout = 'dropout' if args.dropout and args.masking ==0  else 'masking'

if not args.dropout and args.masking == 0:
    dropout = 'no-dropout'

from sklearn.model_selection import KFold
k_fold = 1 if args.k_fold == 0 else args.k_fold
if k_fold == 1:
    splt =[(None,None)]
else:
    kf = KFold(n_splits=k_fold)
    splt = kf.split(train)


directory = f'./results/{args.dataset}/{args.modalities}/{args.task}/{aux}/its:{k_fold}/{dropout}/{args.grouper}/{n_labels}/{epochs}/{batch_size}/ancestry:{args.ancestry}'

if not args.directed:
    directory = f'{directory}/unidirected'
    
if not args.poincare:
    directory = f'{directory}/random'
if args.cooccurrence:
    directory = f'{directory}/cooccurrence'

try:
    saved_results = pd.read_csv(f'./metrics/{".".join(directory.split("/")[1:])}csv',index_col=0)
except FileNotFoundError:
    print("No metrics saved for",directory)
    saved_results = pd.DataFrame()
    
os.makedirs(directory, exist_ok=True)
os.makedirs(directory.replace('results','models'), exist_ok=True)

curr_mx = [0 for i in range(len(args.task))]
idxmax = None
i = 0

project_name = f'Phenotype Prediction-{args.grouper}-{args.task}-{n_labels}'
if k_fold >1:
    project_name = f'{project_name}-{k_fold}-fold'
    
for train_index, test_index in splt:
    print(f'Fold {i+1}')
    i = i + 1
    try:
        train_fold = train.iloc[train_index]
        test_fold = train.iloc[test_index]
    except TypeError:
        train_fold = train
        test_fold = test
    
    x_train,x_validation,y_train,y_validation = data_to_input(train_fold,test_fold)

    training_loader = DataWrapper(x_train,y_train,G=G,data_type=Graph,batch_size=batch_size,masking=masking,replicate=replicate,random=False)


    err = 0
    for embedding_size in embedding_sizes:
        print("Embedding Size",embedding_size)
        embeddings = init_embeddings(embedding_size,poincare = args.poincare,G=G,modalities=args.modalities) 
        if args.ancestry =='no':
            embeddings = embeddings[list(idx.values())]
        validation = get_or_load_data(patients = x_validation,target=y_validation,G = G,embedding_size = embedding_size,graph_type=Graph,embeddings=embeddings,replicate=False,random=False).to(device)
        for n_layer in n_layers:
            if embedding_size == 512 and n_layer >2:
                continue
            f_name = f"SW-Phenotype-{n_layer}-{embedding_size}"
            
            if k_fold > 1:
                f_fame = f'{f_name}-fold:{i}'
                
            if len(args.task) >1:
                f_name = f'{f_name}-causal:{args.causality}'
            print("Embedding size",embedding_size,"N_layer",n_layer)
            print('Running',f_name)
            try:
                saved_results.loc[f_name]
                if not args.override:
                    print("Skipping",f_name)
                    continue
            except:
                pass
            
            if path.exists(f'./{directory}/{f_name}'):
                if not args.override: 
                    print('Skipping',f_name)
                    continue
                else:
                    print('Overriding',f_name)
        

            #config does not need to have emb_sizes and n_layers
            config =  { 'directory':directory,'masking':masking,'batch_size':args.batch_size,'epochs':epochs,'labels':n_labels,"replicate":replicate,**vars(args)}
            run = wandb.init(project=project_name, entity='msd-cardoso',name=f_name+f'-{args.dataset}',config=config)

            model = Net(embedding_size=embedding_size,n_layers=n_layer,output_pathway=n_labels,dropout=args.dropout,causal = args.causality).to(device)
            wandb.watch(model)
            try:
                model_time,loss,epoch= run_batch(model,epochs = epochs,training_loader=training_loader,embedding_size=embedding_size,embeddings=embeddings,device=device,wandb=wandb,validation=validation)
                wandb.log({"model_loss": loss})
            except RuntimeError as msg:
                print(msg)
                run.finish()
                err = err + 1
                if err == 4:
                    break
                continue

            model.eval()
            with torch.no_grad():
                diagnostics_pred = model(validation)
            model.train() #I dont think this is needed here.
            
            #print(f"--Validation-- for {n_layer+1} layers")
            result = {}


            diagnosis = validate(diagnostics_pred,validation.target,wandb=wandb)
            results = diagnosis
            results['epochs'] = epoch
            results['loss'] = loss
            results["time"] = model_time
            df = pd.DataFrame({f_name:results})
           
            if k_fold >1:
           
                for idx,(task,optimize) in enumerate(zip(args.task,args.optimize)):
                    mx = df.T[task].apply(pd.Series)[optimize].max()
                    if mx >= curr_mx[idx]:
                        curr_mx[idx] = mx
                        idxmx = df.T[task].apply(pd.Series)[optimize].astype(float).idxmax()
                        break
                    
            else:  
                torch.save(model.state_dict(), directory.replace('results','models')+f'/{f_name}')
                df.to_pickle(f'./{directory}/{f_name}') 
                print('Saving',directory,f_name)
            print('Run',f_name, "finished")
            run.finish()
            
if k_fold >1:
    print('Evaluating',idxmx)
    splt = idxmx.split('-')
    n_layer = splt[2]
    embedding_size = splt[3]
    model = Net(embedding_size=embedding_size,n_layers=n_layer,output_pathway=n_labels,dropout=masking == 0,causal = args.causality).to(device)
    
    x_train,x_validation,y_train,y_validation = data_to_input(train,test)
    training_loader = DataWrapper(x_train,y_train,G=G,data_type=Graph,batch_size=batch_size,masking=masking,replicate=replicate,random=False)
    embeddings = init_embeddings(embedding_size,poincare = args.poincare,G=G,modalities=args.modalities) 
    validation = get_or_load_data(patients = x_validation,target=y_validation,G = G,embedding_size = embedding_size,graph_type=Graph,embeddings=embeddings,replicate=False,random=False).to(device)
    model_time,loss,epoch= run_batch(model,epochs = epochs,training_loader=training_loader,embedding_size=embedding_size,embeddings=embeddings,device=device,wandb=wandb,validation=validation)
        
    diagnosis = validate(diagnostics_pred,validation.target,wandb=wandb)
    results = diagnosis
    results['epochs'] = epoch
    results['loss'] = loss
    results["time"] = model_time
    df = pd.DataFrame({f_name:results})
    df.to_pickle(f'./{directory}/{f_name}') 
    print('K-fold executed and best result saved.')