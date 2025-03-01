import pandas as pd
import scanpy as sc
import pickle as pk

def load_data(dataset_name):
    subtype = None
    if dataset_name.lower() == 'guo':
        adata = sc.read_csv("../datasets/Guo/GSE99254.tsv", delimiter="\t").T

    elif dataset_name.lower() == "biase":
        adata = sc.read_text(
            "../datasets/Biase/GSE57249_fpkm.txt", delimiter="\t")
        subtype = pd.read_csv("../datasets/Biase/subtype.ann",
                              delimiter="\t")["cell", "label"].values
        celltype_dict = {row[0]: row[1] for row in subtype}
        adata.obs["celltype"] = [celltype_dict.get(
            name, 'Unknown') for name in adata.obs_names]

    elif dataset_name.lower() == 'brown':
        adata = sc.read_csv("../datasets/Brown/hum_melanoma_counts.tsv", delimiter="\t")

    elif dataset_name.lower()== "bjorklund":
        adata = sc.read_csv("../datasets/Bjorklund/Bjorklund.tsv", delimiter="\t")
        subtype = pd.read_csv("../datasets/Bjorklund/labels.ann", delimiter="\t", header=None).rename(columns={0: "cell", 1: "label"})
        # print(subtype)
        celltype_dict = {}
        for i in range(len(subtype)):
            celltype_dict[subtype.iloc[i]["cell"]] = subtype.iloc[i]["label"]
        # print(celltype_dict)
        adata.obs["celltype"] = [celltype_dict.get(
            name, 'Unknown') for name in adata.obs_names]
        # print(adata.obs["celltype"])
        adata = adata[adata.obs["celltype"] != "Unknown",:].copy()
        print(f"Shape of adata: {adata.shape}")
        
        

    elif dataset_name.lower() == 'chung':
        adata = sc.read_csv("../datasets/Chung/GSE75688.tsv", delimiter="\t").T
        subtype = pd.read_csv(
            "../datasets/Chung/GSE75688_final_sample_information.txt", delimiter="\t")
        print(subtype)
        subtype = subtype[["sample", "index3"]].values
        celltype_dict = {row[0]: row[1] for row in subtype}
        adata.obs['celltype'] = [celltype_dict.get(
            name, 'Unknown') for name in adata.obs_names]
        # Filter cells with known celltypes (not 'Unknown')
        known_cells = adata.obs['celltype'] != 'Unknown'
        filtered_adata = adata[known_cells,:]

        # Save expression data
        filtered_adata.to_df().to_csv('../datasets/Chung/Chung_data.csv')

        # Save cell type annotations
        labels = pd.DataFrame({
            'cell': filtered_adata.obs.index,
            'label': filtered_adata.obs['celltype']
        })
        labels.to_csv('../datasets/Chung/label.ann', sep='\t', index=False)

    elif dataset_name.lower() == 'habib':
        habib = sc.read_csv(
            "../datasets/Habib/GSE104525_Mouse_Processed_GTEx_Data.DGE.UMI-Counts.txt", delimiter="\t")
        adata = habib.transpose()

    elif dataset_name.lower() == 'sun':
        adata = pk.load(open("../datasets/Sun/celltype_specific_counts.pkl", "rb"))

    elif dataset_name.lower() == 'pbmc':
        adata = sc.read_10x_mtx("../datasets/pbmc/pbmc6k_matrices",
                                var_names='gene_symbols', cache=False)

    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return adata, subtype


