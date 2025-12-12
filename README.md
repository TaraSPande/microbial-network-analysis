# Microbial Network Analysis
Determining phylogeny of microbial organisms via network topology metrics and functional essentiality methods. Adding a hypothetical Figure 4 to "Identification of critical connectors in the directed reaction-centric graphs of microbial metabolic networks" (Kim et al).

[Final Presentation](https://github.com/TaraSPande/microbial-network-analysis/blob/main/MNA_presentation.pdf)

[All Figures](https://github.com/TaraSPande/microbial-network-analysis/tree/main/figures)

[Network Topology Metrics](https://github.com/TaraSPande/microbial-network-analysis/tree/main/NT-metrics)

### Methods

(1) :: `saveXMLtoCSV` :: convert raw XML models from BiGG database into CSVs

(2) :: `reaction_network_analysis.py` :: main file to run network topology and functional essentiality analysis

(3) :: `tree-similarity.py` :: perform RF distance and Triplets distance on all final tree combinations

### Supplemental
[Network Topology Metric Distributions](https://github.com/TaraSPande/microbial-network-analysis/tree/main/figures/NT-metric-distributions)

(S) :: `NT-metric-distributions.py` :: create NT metric distribution graphs
