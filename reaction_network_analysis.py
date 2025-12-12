"""
Compute bridging centrality and cascade number for reaction-centric directed graphs.
Also: produce a phylogenetic-style heatmap comparing networks based on essential reactions.

#iJO1366.csv iYO844.csv iAF987.csv iYL1228.csv iMM904.csv iCN718.csv iCN900.csv iEK1008.csv iJN678.csv iJN1463.csv iNF517.csv iRC1080.csv iYS854.csv

Usage examples:
    # multiple networks produce clustermap
    python reaction_network_analysis.py \
      --inputs netA.csv netB.csv netC.csv \
      --out-prefix compare

    # if out-prefix_XXXX_metrics.csv already exist:
    python reaction_network_analysis.py \
      --out-prefix compare

The script will treat "essential" reactions as those with cascade_number > 0 (i.e., they trigger cascades).
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA


# --- Core computations -------------------------------------------------------

def load_adjacency_csv(path):
    """
    Load adjacency CSV into a directed NetworkX DiGraph.
    CSV assumed square with row and column labels (reaction IDs).
    Values > 0 are treated as edges.
    """
    df = pd.read_csv(path, index_col=0)
    # ensure columns align
    if list(df.index) != list(df.columns):
        # try to reorder columns to match index if possible
        common = [c for c in df.columns if c in df.index]
        if len(common) == df.shape[0]:
            df = df.loc[df.index, df.index]
        else:
            raise ValueError("Adjacency CSV must have same labels for rows and columns.")
    G = nx.DiGraph()
    nodes = list(df.index.astype(str))
    G.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            val = df.iat[i, j]
            try:
                weight = float(val)
            except:
                continue
            if weight != 0:
                G.add_edge(u, v, weight=weight)
    return G, df

def bridging_coefficient_directed(G):
    """
    Bridging coefficient for directed graph:
    deg(v) = in_deg + out_deg
    N(v) = union of in & out neighbors (as strings)
    BC(v) = (1/deg(v)) / sum_{u in N(v)} (1/deg(u))
    If deg(v) == 0 or denominator == 0, coefficient set to 0.
    Returns dict node -> bridging coefficient.
    """
    bc = {}
    deg = {}
    for n in G.nodes():
        d = G.in_degree(n) + G.out_degree(n)
        deg[n] = d
    for n in G.nodes():
        d = deg[n]
        if d == 0:
            bc[n] = 0.0
            continue
        neigh = set(G.predecessors(n)) | set(G.successors(n))
        denom = 0.0
        for u in neigh:
            du = deg[u]
            if du > 0:
                denom += 1.0/du
        if denom == 0.0:
            bc[n] = 0.0
        else:
            bc[n] = (1.0/d) / denom
    return bc

def bridging_centrality(G):
    """
    Compute betweenness and bridging centrality.
    Returns two dicts: betweenness (directed, normalized) and bridging_centrality.
    """
    # betweenness centrality on directed graph
    betw = nx.betweenness_centrality(G, normalized=True, weight=None)  # weight could be used if meaningful
    bcoef = bridging_coefficient_directed(G)
    brid = {n: bcoef[n] * betw.get(n, 0.0) for n in G.nodes()}
    return betw, bcoef, brid

def cascade_numbers(G):
    """
    For each node r0, simulate removing r0 and iteratively remove reactions
    whose ALL producers (incoming neighbors) have been removed.
    Reactions with zero original in-degree are treated as having external inputs and cannot be removed by cascade.
    Returns dict node -> cascade_number (count of additionally removed nodes).
    """
    original_in = {n: set(G.predecessors(n)) for n in G.nodes()}
    original_indeg = {n: len(original_in[n]) for n in G.nodes()}

    cascade = {}
    nodes_list = list(G.nodes())
    for r0 in nodes_list:
        removed = set([r0])
        # reactions that are 'immune' because they have no producers (external inputs)
        immune = {n for n, d in original_indeg.items() if d == 0}
        # we will not remove immune nodes via cascade
        # iterative propagation
        changed = True
        while changed:
            changed = False
            for n in nodes_list:
                if n in removed or n in immune:
                    continue
                producers = original_in[n]
                # if all producers are removed (or there were none but we handled immune), then n fails
                if producers and producers.issubset(removed):
                    removed.add(n)
                    changed = True
        # cascade number excludes the original node
        cascade[r0] = len(removed) - 1
    return cascade

# --- Multi-network comparison ------------------------------------------------

def build_feature_vector_for_essentials(metrics_df, essential_list):
    """
    From metrics_df (index = nodes) pick rows for essential_list and produce feature vector.
    metrics_df should have columns: 'bridging_coef', 'betweenness', 'bridging_centrality', 'cascade_number'
    We produce summary stats for the set of essential reactions: mean, median, std for each metric,
    concatenated into a vector. This yields fixed-length descriptors per network for clustering.
    
    Safe version: guarantees all finite features.
    """
    # intersect essential_list with actual nodes
    essential_list = [x for x in essential_list if x in metrics_df.index]

    if len(essential_list) == 0:
        # fallback: use top 10 by bridging_centrality
        essential_list = list(
            metrics_df.sort_values('bridging_centrality', ascending=False)
            .head(10).index
        )
    selected = metrics_df.loc[essential_list]

    # replace any non-finite values in selected
    selected = selected.replace([np.inf, -np.inf], np.nan).fillna(0)

    features = []
    for col in ['bridging_coef', 'betweenness', 'bridging_centrality', 'cascade_number']:
        arr = selected[col].values.astype(float)
        if arr.size == 0:
            features.extend([0, 0, 0])
        else:
            features.extend([
                float(np.nanmean(arr)),
                float(np.nanmedian(arr)),
                float(np.nanstd(arr)) if arr.size > 1 else 0.0
            ])

    # Replace any leftover NaNs
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(features)


# --- IO & plotting ----------------------------------------------------------

def compute_and_save_metrics(adj_csv_path, out_prefix):
    G, df = load_adjacency_csv(adj_csv_path)
    betw, bcoef, brid = bridging_centrality(G)
    casc = cascade_numbers(G)
    rows = []
    for n in G.nodes():
        rows.append({
            'node': n,
            'in_degree': G.in_degree(n),
            'out_degree': G.out_degree(n),
            'betweenness': betw.get(n, 0.0),
            'bridging_coef': bcoef.get(n, 0.0),
            'bridging_centrality': brid.get(n, 0.0),
            'cascade_number': casc.get(n, 0)
        })
    metrics_df = pd.DataFrame(rows).set_index('node')
    metrics_df.to_csv(f"{out_prefix}_metrics.csv")
    print(f"Saved metrics to {out_prefix}_metrics.csv")
    return G, metrics_df

def make_heatmap(feature_matrix, labels, outpath):
    """
    feature_matrix: (N_networks, feature_dim) numpy array or DataFrame
    labels: list of network names
    """
    feature_names = [
        "bridging_coef_mean", "bridging_coef_median", "bridging_coef_std",
        "betweenness_mean", "betweenness_median", "betweenness_std",
        "bridging_centrality_mean", "bridging_centrality_median", "bridging_centrality_std",
        "cascade_number_mean", "cascade_number_median", "cascade_number_std"
    ]

    df = pd.DataFrame(feature_matrix, index=labels, columns=feature_names)

    # Drop anything non-numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    df_T = df_scaled.T

    cg = sns.clustermap(
        df_T,
        metric='euclidean',
        method='average',
        cmap='viridis',
        figsize=(10, 8)
    )
    cg.savefig(outpath, dpi=300)
    print(f"Saved heatmap to {outpath}")

def make_phylotree(essential_reactions):
    # Get all reactions across all networks
    all_reactions = sorted(set(r for reactions in essential_reactions.values() for r in reactions))

    # Create binary DataFrame
    df_binary = pd.DataFrame(0, index=essential_reactions.keys(), columns=all_reactions)

    for net, reactions in essential_reactions.items():
        df_binary.loc[net, reactions] = 1

    df_binary.head()

    # pdist expects rows = samples, cols = features
    # metric='jaccard' computes 1 - (|A∩B|/|A∪B|)
    dist_matrix = pdist(df_binary.values, metric='jaccard')
    dist_square = squareform(dist_matrix)  # optional, for inspection

    linkage_matrix = linkage(dist_matrix, method='average')  # UPGMA

    plt.figure(figsize=(10,6))
    dendrogram(linkage_matrix, labels=df_binary.index, leaf_rotation=90)
    plt.title("Phylogeny of microbial networks based on essential reactions")
    plt.ylabel("Jaccard distance")
    plt.show()

    # Convert distance to similarity (optional)
    sim_matrix = 1 - squareform(dist_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=df_binary.index, columns=df_binary.index)

    sns.clustermap(sim_df, cmap="viridis", figsize=(8,6))
    plt.show()

def make_pca(feature_matrix, labels):
    # Convert to dataframe
    df = pd.DataFrame(feature_matrix, index=labels)

    X_scaled = StandardScaler().fit_transform(feature_matrix)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X_scaled)

    #Color by Phylum
    phylum_map = {
        "iAF987": "Pseudomonadota",
        "iCN718": "Pseudomonadota",
        "iCN900": "Bacillota",
        "iEK1008": "Actinomycetota",
        "iJN678": "Cyanobacteriota",
        "iJN1463": "Pseudomonadota",
        "iJO1366": "Pseudomonadota",
        "iMM904": "Ascomycota (E)",
        "iNF517": "Bacillota",
        "iRC1080": "Chlorophyta (E)",
        "iYL1228": "Pseudomonadota",
        "iYO844": "Bacillota",
        "iYS854": "Bacillota"
    }
    phylum = [phylum_map[lbl] for lbl in labels]
    custom_palette = {
        'Pseudomonadota': '#EF8535',
        'Bacillota': '#519D40', 
        'Actinomycetota': '#3A75B2',
        'Cyanobacteriota': '#C23A30',
        'Ascomycota (E)': '#8E68B8',
        'Chlorophyta (E)': '#8E68B8'
    }

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            x=embedding[:,0],
            y=embedding[:,1],
            hue=phylum,
            palette=custom_palette,
            s=80)

    # Add labels
    for label, x, y in zip(labels, embedding[:, 0], embedding[:, 1]):
        plt.text(x + 0.02, y + 0.02, label)

    plt.title("PCA of Organism Network Topologies")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()

    return embedding


# --- Command-line interface -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute bridging centrality and cascade numbers; compare networks.")
    parser.add_argument('--input', dest='input', help='Single adjacency CSV input', type=str)
    parser.add_argument('--inputs', dest='inputs', nargs='*', help='Multiple adjacency CSVs for comparison', type=str)
    parser.add_argument('--out-prefix', dest='out_prefix', type=str, default='result')
    args = parser.parse_args()

    if args.input:
        compute_and_save_metrics(args.input, args.out_prefix)
        return

    metrics = {}
    if args.inputs:
        nets = {}
        for p in args.inputs:
            name = Path(p).stem
            G, df_metrics = compute_and_save_metrics(p, f"{args.out_prefix}_{name}")
            nets[name] = G
            metrics[name] = df_metrics
    else:
        model_names = ["iJO1366", "iYO844", "iAF987", "iYL1228", "iMM904", "iCN718", "iCN900", "iEK1008", "iJN678", "iJN1463", "iNF517", "iRC1080", "iYS854"]
        for p in model_names:
            metrics[p] = pd.read_csv(f"compare_{p}_metrics.csv", index_col=0)

    # Build feature vectors for each network
    labels = []
    feats = []
    essential_reactions = {} # Example: dictionary of networks → essential reactions
    for name, dfm in metrics.items():
        labels.append(name)
        # treat "essential" as nodes with cascade_number > 0
        ess = list(dfm.index[dfm['cascade_number'] > 0])
        essential_reactions[name] = ess
        if len(ess) == 0:
            # fallback: top 10 nodes by bridging_centrality
            ess = list(dfm.sort_values('bridging_centrality', ascending=False).head(10).index)
        feats.append(build_feature_vector_for_essentials(dfm, ess))
    feature_matrix = np.vstack(feats)

    # Make clustermap and pca (NT)
    make_heatmap(feature_matrix, labels, f"{args.out_prefix}_clustermap.png")
    make_pca(feature_matrix, labels)
    # Save pairwise distance matrix
    dist = squareform(pdist(feature_matrix, metric='euclidean'))
    pd.DataFrame(dist, index=labels, columns=labels).to_csv(f"{args.out_prefix}_pairwise_distances.csv")
    print(f"Saved pairwise distance matrix to {args.out_prefix}_pairwise_distances.csv")

    # Make phylogenetic tree and clustermap (FE)
    make_phylotree(essential_reactions)

    return

    parser.print_help()

if __name__ == "__main__":
    main()
