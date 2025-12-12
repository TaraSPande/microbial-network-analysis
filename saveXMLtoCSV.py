from libsbml import readSBML
import numpy as np
import pandas as pd


model_names = ["iJO1366", "iYO844", "iAF987", "iYL1228", "iMM904", "iCN718", "iCN900", "iEK1008", "iJN678", "iJN1463", "iNF517", "iRC1080", "iYS854"]
for mn in model_names:
    print("Building " + mn + " model...")
    doc = readSBML(mn + ".xml")
    model = doc.getModel()

    # Get reactions
    reactions = model.getListOfReactions()
    rxn_ids = [r.getId() for r in reactions]
    n = len(reactions)
    A = np.zeros((n, n), dtype=int)
    idx = {rid: i for i, rid in enumerate(rxn_ids)}

    # Build metabolite → (producers, consumers)
    produces = {}
    consumes = {}

    for r in reactions:
        rid = r.getId()
        for reactant in r.getListOfReactants():
            mid = reactant.getSpecies()
            consumes.setdefault(mid, []).append(rid)
        for product in r.getListOfProducts():
            mid = product.getSpecies()
            produces.setdefault(mid, []).append(rid)

    # Build adjacency
    for m in produces:
        if m not in consumes:
            continue
        for r1 in produces[m]:
            for r2 in consumes[m]:
                A[idx[r1], idx[r2]] = 1

    df = pd.DataFrame(A, index=rxn_ids, columns=rxn_ids)
    df.to_csv(mn + ".csv")
