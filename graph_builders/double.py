from .base import BaseGraph
import torch


#DIAGNOSES NEED TO BE FIRST AND PROCEDURES SECOND
class DoubleGraph(BaseGraph):
   
     #TODO: Move implementation towards diagnoses.py
    
    def add_patient(x,patient,i,instance=None,lookup=None,random=None,**kwargs):
        instance.add_node(f"Patient_{i}")
        x = torch.cat((x,torch.zeros(1,x.shape[1]))) 
        visits = []
        v = 0
        #for each visit..
        for visit in patient:
            
            visit_name = f'{i}_Visit_{v}'
            
            x = torch.cat((x,torch.zeros(1,x.shape[1]))) 
            visits.append(len(instance.nodes))
            instance.add_node(visit_name)
            instance.add_edge(visit_name,visit_name)   
         
        
            for code in visit[0]:
                assert f'D{code}' in instance.nodes,f'D{code} does not exist in the graph'
                instance.add_edge(f'D{code}',visit_name)
            for code in visit[1]:
                assert f'P{code}' in instance.nodes,f'P{code} does not exist in the graph'
                instance.add_edge(f'P{code}',visit_name)


            v = v + 1
        for visit in range(0,v):
            for future_visit in range(visit+1,v):
                instance.add_edge(f'{i}_Visit_{visit}',f'{i}_Visit_{future_visit}')
                
        return x,visits
