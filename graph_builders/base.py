from abc import ABC, abstractmethod
import torch
import networkx as nx

class BaseGraph(ABC):
       
    
    @classmethod
    def create_graph(cls,G=None,patients=None,embedding_size=None,embeddings = None,replicate=False,**kwargs):
   
        i=0
        nodes = []
        instance = G.copy()
         
        patients_visits = {}
        output_nodes = []
        if embeddings != None:
            x = embeddings
        else:
            x = torch.empty((0,embedding_size))
        for patient in patients:
            x,visits = cls.add_patient(x,patient,i,instance=instance,**kwargs)
            patients_visits[i] =  visits  #Last one doesnt exist.
            if replicate:
                output_nodes = output_nodes + visits #All visit nodes are output nodes.
                #print(len(visits),len(output_nodes))
            else:
                output_nodes.append(visits[-1])
              
            i = i + 1
        
        assert len(instance.nodes) ==len(x), f"{len(instance.nodes)} does not match {len(x)}"
        edge_index = torch.tensor(list(nx.convert_node_labels_to_integers(instance).edges)).t().contiguous()
        kwargs = {"output_nodes":output_nodes,"edge_index":edge_index,"patient":patients_visits,**kwargs}
        #print(f"Graph created with {len(nodes)} patient nodes and a total of {len(instance.nodes)} nodes")
        return x,kwargs
    
    @staticmethod
    @abstractmethod
    def add_patient(*args,**kwargs):
        pass
    
  
  