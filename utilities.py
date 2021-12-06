from metrics import *
import torch
from torch_geometric.data import Data
import time
import numpy as np

def validate(y_pred,y_real,ks=[5,10,15,20,25,30],wandb = None):
    pred_size = len(y_pred)
    
    tasks = list(y_real.keys())
    y_real = list(y_real.values())
   
    all_results ={}
    for item in range(pred_size):
        results = {}
        for k in ks:
            metrics = calculate_metrics(y_pred[item],y_real[item],k=k)
            results = {**results,**metrics}
        all_results[tasks[item]]= results
    if wandb is not None:
        wandb.log(all_results)
    #print(all_results)
    return all_results



def run_batch(model,epochs=15,training_loader=None,device=None,wandb=None,validation=None,**kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00045)
    loss_fn = torch.nn.BCELoss() 
    print(f"Training for {epochs} epochs")
    losses = []
    start = time.time()
  
    for epoch in range(0,epochs):
        epoch_start = time.time()
        print("Starting epoch",epoch+1)
        training_loader.reset()
        i = 0
        loss_sum = 0
        while True:
            nxt = training_loader.next(**kwargs)
            if not nxt:
                break
            nxt = nxt.to(device)
            out = model(nxt)
            y_real = nxt.target

            if len(out) != len(y_real):
                print(f'Output size and expected target size does not match {len(out)} != {len(y_real)}')
                assert len(out) <= len(y_real),"Output size must be less or equal than expected size" #???
            pred_size = len(out)
            loss = 0

            for item in range(pred_size):
                loss = loss + loss_fn(out[item],list(y_real.values())[item]) #weighted loss?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.cpu().detach().numpy()
            wandb.log({'batch_size':len(nxt.patient),'batch':i,'batch_loss':loss.cpu().detach().numpy()})
            i = i + 1
        
        losses.append(loss_sum/i)
        wandb.log({"epoch_loss": loss_sum/i,"epoch":epoch,"epoch_time":(time.time()-epoch_start)})
        print("Epoch",epoch+1,"Took",(time.time()-epoch_start),"seconds","Loss:", loss_sum/i)
        #TODO
        #model eval
        model.eval()
        with torch.no_grad():
            diagnostics_pred = model(validation)
            diagnosis = validate(diagnostics_pred,validation.target,wandb=wandb)
        model.train()
    model_time = time.time()-start
    wandb.log({"time":model_time})
    print(f'Took {model_time}')
    return model_time,losses,epoch



#
def get_or_load_data(target:dict=None,override:bool=False,graph_type = None,G=None,**kwargs):
 
    assert G != None, "No base graph present. Please add G"
    
    assert type(target) is dict
    try:
        replicate = kwargs["replicate"]
    except KeyError:
        replicate = False
    x,kwargs = graph_type.create_graph(G=G,**kwargs)
   
    if replicate:
        to_tensor = lambda x:torch.tensor(np.concatenate(x)).float()
    else: 
        to_tensor = lambda x:torch.tensor(x).float()

    if target is None:
        return Data(x=x,**kwargs)
    if type(target) is dict: 
        for key in target.keys():
            if target[key].dtype is np.dtype('object') and not replicate:
                #print([len(i[-1]) for i in target[key]])
                target[key] = np.array([i[-1] for i in target[key]])
               
            target[key] = to_tensor(target[key])
    else:
        assert 'Something went wrong.'
    
    
    return Data(x=x,**kwargs,target =target)



#REVIEW
class CustomLoader:
    def __init__(self,x,target,batch_size = 64):
        self.x = x
        self.target = target
        self.iteration = 0
        self.batch_size = batch_size
     


    def next(self):
        mx = np.ceil(len(self.x)/self.batch_size)
        if mx == self.iteration:
            return None
        
        slc = slice(self.iteration*self.batch_size,(self.iteration+1)*self.batch_size)
        x_batch = self.x[slc]
        if type(self.target) is dict:
            y_batch = {}
            for key in self.target.keys():
                y_batch[key] = self.target[key][slc]
        else:
            y_batch = self.target[slc]
        self.iteration = self.iteration + 1
        
        return x_batch,y_batch
    
    def reset(self):
        #add timestep thing
        if type(self.target) is dict:
            for key in self.target.keys():
                assert len(self.x) == len(self.target[key])
        else:
            assert len(self.x) == len(self.target)
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        
        if type(self.target) is dict:
            for key in self.target.keys():
                self.target[key] = self.target[key][p]
        else:
            self.target = self.target[p]
        self.iteration = 0
        
        
    
class DataWrapper:
    def __init__(self,x,target,G= None,batch_size=64,data_type = None,**kwargs):
        self.loader = CustomLoader(x,target,batch_size = batch_size)
        self.G = G 
        self.data_type = data_type
        self.kwargs = kwargs
        
    def next(self,embedding_size=None,embeddings=None,lookup=None):
        nxt = self.loader.next()
        if not nxt:
            return None
        x_batch = nxt[0]
        y_batch = nxt[1]
        return get_or_load_data(G= self.G,patients=x_batch, target=y_batch,embedding_size = embedding_size,embeddings = embeddings,lookup=lookup,graph_type=self.data_type,**self.kwargs)
        
    def reset(self):
        self.loader.reset()

        
def to_pair(diagnoses,procedures):
    res = []
    for (item1,item2) in zip(diagnoses,procedures):
        temp = []
        for(item_a,item_b) in zip(item1,item2):
            temp.append([item_a,item_b])
        res.append(temp)
    return np.array(res)

def data_to_input(train,test,modalities:list = ["diagnoses"],tasks:list=['phenotype']):

    if len(modalities) == 2:
        diagnoses_train = train.diagnoses.values
        procedures_train = train.procedures.values

        diagnoses_validation = test.diagnoses.values
        procedures_validation = train.procedures.values
        x_train = to_pair(diagnoses_train,procedures_train)
        x_validation = to_pair(diagnoses_validation,procedures_validation)

    else:
        x_train = train[modalities].values
        x_validation = test[modalities].values

    y_train = {}
    y_validation ={}
    for task in tasks:
        dim = len(np.array(list(train[task]),dtype=object).shape) #this should be deprecated
        y_train[task] = np.array(list(train[task]))
        y_validation[task] = np.array(list(test[task]))
        #Kinda hacky move.. nested arrays 'are' 1-shaped..
        if dim == 1 and y_validation[task].dtype is not np.dtype('object'):
            y_train[task] = y_train[task].reshape(-1,1)
            y_validation[task] = y_validation[task].reshape(-1,1)
            
    return x_train,x_validation,y_train,y_validation
