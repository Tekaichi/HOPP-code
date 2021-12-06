import torch
from torch import nn

from torch_geometric.nn import GATConv
from torch.functional import F

Vector = list[int]

class OutcomeModel(torch.nn.Module):
    def __init__(self,embedding_size:int=256,n_layers:int = 1,output_pathway:Vector=[283],dropout=True,causal=True):
        super(OutcomeModel, self).__init__()
        
        heads = 8
        self.return_pathway = output_pathway is not None
        self.initial_layer = GATConv(embedding_size,embedding_size,heads=heads)
        self.causal = causal
        #Shared weights model
        self.layers = nn.ModuleList(
          [
            GATConv(embedding_size*heads,embedding_size,heads=heads)
           
            for _ in range(n_layers)
            ]
        )
        
        self.leaky = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(p=0.2) if dropout else lambda x: x

        
        def get_previous(classes,idx):
            if idx == 0 or not causal:
                return 0
            return classes[idx-1]
                
        
        
        self.pathway = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embedding_size*heads+get_previous(output_pathway,idx),output),
                nn.Sigmoid(),
                )
                if output == 1 else 
                nn.Sequential(
                nn.Linear(embedding_size*heads+get_previous(output_pathway,idx),output),
                nn.Softmax(dim=1),
                )
                for idx,output in enumerate(output_pathway)
            ]
        )
            
                
    def forward(self, data,output=None):
        
        x = data.x

        edge_index = data.edge_index
        
        x,(edge_index,attention_weights) = self.initial_layer(x,edge_index,return_attention_weights=True)
        x = self.leaky(x)

        for layer in self.layers:
            x,(edge_index,attention_weights) = layer(x,edge_index,return_attention_weights=True)
            x = self.dropout(x)
            x = self.leaky(x)
        
        self.x = x        
        self.attention = (edge_index,attention_weights)
        
        output_nodes = data.output_nodes
        
        def causal(prev,current):
            if self.causal:
                return torch.cat((current,prev),axis=1)
            return current
        
        results = []
        idx = 0
        if output == None:
            results.append(self.pathway[0](x[output_nodes]))
        else:
            assert output.shape[0] == 283, 'Model not prepared to process multiple interactions simultaneously'
            output = output.view(1,-1)
            results.append(output)
            
        for pathway in self.pathway[1:]:
            results.append(pathway(causal(results[idx],x[output_nodes])))
            idx = idx + 1
        return results
      