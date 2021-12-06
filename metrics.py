import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
def top_diseases(y_pred,classes):
    y_pred = y_pred.mean(dim=0).argsort(descending=True)
    return torch.tensor(classes).gather(0,y_pred)

def recall(y_pred,y_real,k=10):
    sort = y_pred.sort(axis=1,descending=True)
    y_pred = sort.indices[:,0:k]
    top_k = y_real.gather(1,y_pred)

    return top_k.sum(dim=1)/torch.clamp(y_real.sum(dim=1),max=k)
def calculate_metrics(y_pred,y_real,k=10,axis=None):
    

        
    #Remove those that have no positive label.
    #TODO if no positive labei is present, the prediction should be equally distributed among all classes?
    if axis is None:
        axis = 0 if y_pred.shape[1] == 1 else 1
    if axis == 1:
        y_pred = y_pred[y_real.sum(dim=1)!=0]
        y_real = y_real[y_real.sum(dim=1)!=0]
    
    if y_pred.shape[1] ==1:
        return {**fnrate(y_pred,y_real),'died':y_real.sum().cpu().detach().numpy().item(),'accuracy':get_accuracy(y_pred.round().cpu().detach().numpy(),y_real.cpu().detach().numpy())}
    sort = y_pred.sort(axis=axis,descending=True)
    
    if axis == 1:
        y_pred = sort.indices[:,0:k]
        top_k = y_real.gather(axis,y_pred)

    else:
        y_pred = sort.indices[:k]
        top_k = y_real[y_pred].flatten()

    
   
    div = torch.clamp(y_real.sum(dim=axis),max=k)
    
               
    recall = (top_k.sum(dim=axis)/div).mean()
    precision = (top_k.sum(dim=axis)/k)
    precision = precision.mean()

    return {f"R@{k}":recall.cpu().detach().numpy(),f"P@{k}":precision.cpu().detach().numpy()}


#false negative rate | type 2 error
def fnrate(y_pred,y_real):
    y_pred = y_pred.cpu().detach().numpy()
    y_real = y_real.cpu().detach().numpy()
    c_matrix = confusion_matrix(y_real, y_pred.round(),labels=[0,1])
    tn, fp, fn, tp  =c_matrix.ravel()
    #return {"False Negative Rate":fn/(tp+fn),
    return {"sensivity":    tp / (tp + fn), "tp":tp,"fn":fn,"fp":fp,"tn":tn,"specificity":tn/(tn+fp)}
def get_accuracy(y_pred,y_real):
    #y_pred = y_pred.detach().numpy()
    #y_real = y_real.detach().numpy()
    return accuracy_score(y_real,y_pred)

