import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model_names = ["iJO1366", "iYO844", "iAF987", "iYL1228", "iMM904", "iCN718", "iCN900", "iEK1008", "iJN678", "iJN1463", "iNF517", "iRC1080", "iYS854"]

dfs = []
labels = []
for mn in model_names:
	path = f"compare_{mn}_metrics.csv"
	df = pd.read_csv(path, index_col=0)
	dfs.append(df)
	labels.append(mn)

numeric_cols = dfs[0].select_dtypes(include=['float64', 'int64']).columns

colors = sns.color_palette("tab20", n_colors=len(dfs))  # nice color palette

for col in numeric_cols:
    plt.figure(figsize=(8,5))
    for df, label, color in zip(dfs, labels, colors):
        sns.histplot(df[col], kde=False, bins=20, alpha=0.5, label=label, color=color, stat="density")
    
    plt.title(f"Overlapping Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    all_data = pd.concat([df[col] for df in dfs])
    plt.xlim(all_data.min(), all_data.max())

    plt.legend()
    plt.show()
