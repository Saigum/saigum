import torch
import scanpy as sc
from model import maskedDataset, MaskedAutoencoder
from scipy.sparse import csr_matrix, issparse
import pandas as pd
import argparse 
from data_utils import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr

## load data

def print_data_details(data):
    print(data)
    print(data.shape)
    print(data.dtype)

def print_adata_details(adata, subtypes):
    print(adata)
    print(subtypes)
    print(f"Shape of adata: {adata.shape}")
    print(f"Statistics of adata.X: mean={adata.X.mean():.4f}, std={adata.X.std():.4f}, min={adata.X.min():.4f}, max={adata.X.max():.4f}")
    print(f"Shape of subtypes: {subtypes.shape}")

def run_args(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    
    ################################################################################################
    ############# Single gene expression data (only one dataset gene reconstruction ################
    ################################################################################################
    
    print("=============================================")
    print("Running Single Dataset Reconstruction Testing ")
    print("=============================================\n")
    adata, subtypes = load_data(args.data)
    
    ### Data Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    print_adata_details(adata, subtypes)
    
    data = csr_matrix(adata.X) if issparse(adata.X) else adata.X
    data = torch.tensor(data)
    print_data_details(data)
    
    # Create dataset with masking
    dataset = maskedDataset(data, mask_percent=args.mask_percent)
    train_data, val_data = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    model = MaskedAutoencoder(input_dim=data.shape[1], hidden_dim=[512,256,128], latent_dim=args.latent_dim).to(device)
    
    ######################################################
    ############### Training Loop ########################
    ######################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer
    model.train()
    loss_log = []
    val_loss_log = []
    
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                model.zero_grad()
                x = x.float()
                y = y.float()
                x_hat, latent = model(x.to(device))
                loss = torch.nn.functional.mse_loss(x_hat, y.to(device))
                loss.backward()
                model.optimizer.step()
                loss_log.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.float()
                    y = y.float()
                    x_hat, latent = model(x.to(device))
                    loss = torch.nn.functional.mse_loss(x_hat, y.to(device))
                    val_loss_log.append(loss.item())
                    
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch}, Last Loss: {loss.item():.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(loss_log, label="Train Loss")
    plt.plot(val_loss_log, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(args.plot_path)
    plt.close()
    
    ###########################################################
    #################### Evaluation ###########################
    ###########################################################
    
    # New random mask applied to data for testing
    test_dataset = maskedDataset(data, mask_percent=args.mask_percent)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_loss = 0
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float()
            y = y.float()
            x_hat, latent = model(x.to(device))
            loss = torch.nn.functional.mse_loss(x_hat, y.to(device))
            test_loss += loss.item()
            # Save predictions and targets (move to CPU and numpy for metrics)
            all_preds.append(x_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    ## evaluate over masked genes only (not the whole dataset)
    masked_preds = all_preds
    masked_targets = all_targets
    masked_preds = masked_preds[dataset.mask]
    masked_targets = masked_targets[dataset.mask] 
    
    print(f"Sanity check, masked_preds shape: {masked_preds.shape}, true masked_targets shape: {masked_targets.shape}")
    
    rmse = np.sqrt(mean_squared_error(masked_preds,masked_targets ))
    mae = mean_absolute_error(masked_targets, masked_preds)
    overall_corr, _ = pearsonr(masked_preds.flatten(), masked_targets.flatten())
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Overall Pearson Correlation: {overall_corr:.4f}")
    torch.save(model.state_dict(), args.output_path)
    
    ### Evaluating prediction when you mask the same set of genes
    #############################################################################################################
    ###############  Multiple gene expression data (multiple datasets gene reconstruction)  #####################
    #############################################################################################################
    
    
    
    
    
    
    
    
    
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--mask_percent', type=float, default=0.3)
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
