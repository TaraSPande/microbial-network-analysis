from ete3 import Tree
import dendropy
from dendropy.calculate import treecompare


# Trees in Newick format
GT = "((iMM904, iRC1080), ((((iYL1228, iJO1366), iJN1463, iCN718), iAF987), (iEK1008, iJN678, (iCN900, (iNF517, (iYO844, iYS854))))));"
NT = "(iNF517, (((iCN718, (iJN678, (iAF987, iEK1008))), (iMM904, (iCN900, iRC1080))), ((iJN1463, (iJO1366, iYL1228)), (iYO844, iYS854))));"
FE = "(iRC1080, (iMM904, (iYS854, ((iJN1463, ((iJO1366, iYL1228), (iAF987, iJN678))), (iNF517, (iCN900, (iEK1008, (iYO844, iCN718))))))));"

####ROBINSON-FOULDS DISTANCE (repeat for all combos)####

t1 = Tree(GT)
t2 = Tree(NT)
t3 = Tree (FE)

# Compute RF distance
rf_result = t1.robinson_foulds(t2, unrooted_trees=True)

rf_distance = rf_result[0]      # RF distance
max_rf = rf_result[1]           # Maximum possible RF distance
common_edges = rf_result[2]     # Number of common edges

normalized_rf = rf_distance / max_rf

print(f"RF distance: {rf_distance}")
print(f"Max RF: {max_rf}")
print(f"Normalized RF: {normalized_rf:.3f}")
print(f"Common edges: {common_edges}")


####TRIPLET DISTANCE####

trees_newick = [GT, NT, FE]

# Create a single TaxonNamespace for all trees
taxa = dendropy.TaxonNamespace()

# Parse all trees with the same TaxonNamespace
trees = [dendropy.Tree.get(data=t, schema="newick", taxon_namespace=taxa) for t in trees_newick]

# Compute pairwise symmetric differences (triplet-like distances)
for i in range(len(trees)):
    for j in range(i+1, len(trees)):
        dist = treecompare.symmetric_difference(trees[i], trees[j])
        print(f"Triplet distance between tree {i} and tree {j}: {dist}")

