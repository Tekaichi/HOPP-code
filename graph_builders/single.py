from .base import BaseGraph
import torch
import random

# Node masking here?

class SingleGraph(BaseGraph):
   

    def add_patient(x,patient,i,instance= None,lookup=None,masking_probability=0.0,bidirectional = False,**kwargs):
        
        visits = []
        use_random = kwargs.pop("random",False) 
        if use_random:
            base_tensor = lambda x: torch.rand(1,x.shape[1])
        else:
            base_tensor = lambda x: torch.zeros(1,x.shape[1])
        v = 0
        for visit in patient:
            visit_name = f'{i}_Visit_{v}'
            x = torch.cat((x,base_tensor(x))) #TODO: Initialize with current age
            visits.append(len(instance.nodes))
            instance.add_node(visit_name)
            instance.add_edge(visit_name,visit_name)

            for code in visit:
                #if random less than prob value, 'mask node'
                #e.g if 0.2 and prob is 0.3 then mask the node
                if random.uniform(0,1) < masking_probability:
                    continue
                if code not in instance.nodes:
                    code_embedding = None
                    if lookup:
                        code_embedding = lookup[code].view(1,-1)
                    else:
                        code_embedding = torch.rand(1,x.shape[1])
                    x = torch.cat((x,code_embedding))
                #assume that the same code is not given twice in the same visit..
                instance.add_edge(code,visit_name)

            v = v + 1
        

        #----
        for visit in range(0,v):
            for future_visit in range(visit+1,v):
                instance.add_edge(f'{i}_Visit_{visit}',f'{i}_Visit_{future_visit}')
        #if bidirectional v(t+1) -> v(t) && v(t) -> v(t+1)
        if bidirectional:
            for visit in range(0,v):
                for future_visit in range(visit+1,v):
                    instance.add_edge(f'{i}_Visit_{future_visit}',f'{i}_Visit_{visit}')
        return x,visits
