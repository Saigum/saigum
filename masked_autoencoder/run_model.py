import torch
import scanpy as sc
from model import maskedDataset,MaskedAutoencoder
from scipy.sparse import csr_matrix,issparse
import pandas as pd
import argparse 
from data_utils import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt

## load data

def print_data_details(data):
    print(data)
    print(data.shape)
    print(data.dtype)

def print_adata_details(adata,subtypes):
    print(adata)
    print(subtypes)
    print(f"Shape of adata: {adata.shape}")
    print(f"Statistics of adata.X: mean={adata.X.mean():.4f}, std={adata.X.std():.4f}, min={adata.X.min():.4f}, max={adata.X.max():.4f}")
    print(f"Shape of subtypes: {subtypes.shape}")
    
def run_args(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata,subtypes = load_data(args.data)
    print_adata_details(adata,subtypes)
    data = csr_matrix(adata.X) if issparse(adata.X) else adata.X
    data = torch.tensor(data)
    print_data_details(data)
    dataset = maskedDataset(data,mask_percent=args.mask_percent)
    train_data,val_data = torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True)
    val_data = torch.utils.data.DataLoader(val_data,batch_size=32,shuffle=True)
    model = MaskedAutoencoder(input_dim = data.shape[1],hidden_dim=[512,256,128],latent_dim=64).to(device)
    
    ######################################################
    ############### Training Loop ########################
    ######################################################
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    model.optimizer = optimizer
    model.train()
    loss_log=[]
    val_loss_log=[]
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            for x,y in train_loader:
                model.zero_grad()
                x = x.float().to(device)
                y = y.float().to(device)
                x_hat = model(x)
                loss = torch.nn.functional.mse_loss(x_hat,y)
                loss.backward()
                model.optimizer.step()
                loss_log.append(loss.item())
            model.eval()
            with torch.no_grad():    
                for x,y in val_data:
                    x = x.float()
                    y = y.float()
                    x_hat = model(x)
                    loss = torch.nn.functional.mse_loss(x_hat,y)
                    val_loss_log.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        plt.plot(loss_log,label="train_loss")
        plt.plot(val_loss_log,label="val_loss")
        plt.legend()
        plt.savefig("loss_plot.png")
        torch.save(model.state_dict(),args.output_path)
    

        
        
    
    
    





def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--mask_percent', type=float ,default= 0.3)
    parser.add_argument('--output_path', type=str, default="output.h5ad")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=str, nargs='+', default="[512,256,128]")
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--plot_path', type=str, default="loss_plot.png")

    args = parser.parse_args()
    print(args)
    run_args(args)
    
  
if __name__ == "__main__":
    __main__()
