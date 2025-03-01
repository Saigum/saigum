import torch 
import scanpy as sc



class maskedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 mask_percent:float = 0.3,
                 method:str = "normal"):
        self.data = data
        self.mask_percent = mask_percent
        self.method = method
        self.masked_data = self.mask_data()
    def mask_data(self):#
        if(self.method=="normal"):
            mask = torch.rand(self.data.shape) < self.mask_percent
            masked_data = self.data.clone()
            masked_data[mask] = -1
            self.mask = mask    
        return masked_data
    def __getitem__(self, index):
        return (self.masked_data[index],self.data[index])
    def __len__(self):
        return len(self.data)



# class multiDatamaskedDataset(torch.utils.data.Dataset):
#     ''' Dataset class to handle multiple datasets '''
#     def __init__(self,adata_list,
#                  mask_percent: float = 0.3,
#                  method:str = "normal",
#                  mode:str= "interleave",
#                  ):
#         self.adata_list = adata_list
#         self.mask_percent = mask_percent
#         self.method = method
    
#     def __getitem__(self, index):
        
        
    
    


class Encoder(torch.nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: list = [512,256,128],
                 latent_dim: int =64,
                 dropout: float= 0.3):
        super(Encoder,self).__init__()
        layer_list = hidden_dim.copy()
        layer_list.append(latent_dim)
        layer_list.insert(0, input_dim)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer_list[i], layer_list[i+1])
                for i in range(len(layer_list)-1)
            ]
        )
        # Create BatchNorm layers for each hidden layer
        self.batch_norms = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(layer_list[i+1])
                for i in range(len(layer_list)-2)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1:
                x = torch.nn.functional.relu(x)
                x = self.batch_norms[i](x)
                if self.training:
                    x = self.dropout(x)
        return x
        
    
class Decoder(torch.nn.Module):
    def __init__(self, 
                 latent_dim: int,
                 output_dim: int,
                 hidden_dim: list = [128, 256, 512],
                 dropout: float = 0.3):
        super(Decoder,self).__init__()
        layer_list = hidden_dim.copy()
        layer_list.append(output_dim)
        layer_list.insert(0, latent_dim)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer_list[i], layer_list[i+1])
                for i in range(len(layer_list)-1)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1:
                x = torch.nn.functional.relu(x)
                if(self.training):
                    x = self.dropout(x)
        return x
        
class MaskedAutoencoder(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: list = [512, 256, 128],
                 latent_dim: int = 64,
                 dropout: float = 0.3):
        super(MaskedAutoencoder,self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim, dropout)
    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return x,encoded
    
# class GraphAttentionMaskedAutoencoder(torch.nn.Module):


