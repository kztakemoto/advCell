#!/usr/bin/env python3
"""
Extract only Tabula Sapiens v2 from the Census,
restricting the query to the Tabula Sapiens collection instead of the whole Census.
"""

import pandas as pd
import cellxgene_census

print("=" * 70)
print("Step 1: identify the Tabula Sapiens dataset_ids")
print("=" * 70)

with cellxgene_census.open_soma(census_version="2025-11-08") as census:
    # first fetch the information on all datasets
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    
    # search for datasets whose collection contains "Tabula Sapiens"
    ts_datasets = datasets[
        datasets['collection_name'].str.contains('Tabula Sapiens', case=False, na=False)
    ]
    
    print(f"Number of Tabula Sapiens datasets: {len(ts_datasets)}")
    print(f"\nCollection names:")
    for name in ts_datasets['collection_name'].unique():
        print(f"  {name}")
    
    print(f"\nDetails of each dataset:")
    for _, ds in ts_datasets.iterrows():
        print(f"  dataset_id: {ds['dataset_id']}")
        print(f"    title: {ds.get('dataset_title', 'N/A')}")
        print(f"    collection: {ds['collection_name']}")
        print(f"    cell_count: {ds.get('dataset_total_cell_count', 'N/A')}")
        print()
    
    # build the list of dataset_ids
    ts_dataset_ids = ts_datasets['dataset_id'].tolist()
    
    # filter by dataset_id and fetch the metadata
    print("=" * 70)
    print("Step 2: fetch only the Tabula Sapiens metadata")
    print("=" * 70)
    
    # filter by dataset_id
    filter_str = "dataset_id in ['" + "','".join(ts_dataset_ids) + "']"
    print(f"Filter: {filter_str[:100]}...")
    
    obs_df = census["census_data"]["homo_sapiens"].obs.read(
        value_filter=filter_str,
        column_names=[
            "cell_type", "tissue", "tissue_general",
            "dataset_id", "donor_id", "assay",
            "cell_type_ontology_term_id",
            "is_primary_data",
        ]
    ).concat().to_pandas()
    
    print(f"\nTabula Sapiens total: {len(obs_df)} cells")
    
    # distinguish v1 and v2 (based on donor_id)
    print(f"\n=== Donor distribution ===")
    donor_counts = obs_df.groupby('donor_id').size().sort_values(ascending=False)
    for d, c in donor_counts.items():
        print(f"  {d}: {c} cells")
    
    # filter by is_primary_data (remove duplicates)
    if 'is_primary_data' in obs_df.columns:
        primary = obs_df[obs_df['is_primary_data'] == True]
        print(f"\nis_primary_data=True: {len(primary)} cells")
        print(f"is_primary_data=False: {len(obs_df) - len(primary)} cells")
    
    # cell-type summary
    print(f"\n{'='*70}")
    print("Step 3: cell-type inventory")
    print(f"{'='*70}")
    
    ct_summary = obs_df.groupby('cell_type').agg(
        n_cells=('cell_type', 'size'),
        n_tissues=('tissue', 'nunique'),
        n_donors=('donor_id', 'nunique'),
        tissues=('tissue', lambda x: ', '.join(sorted(x.unique()))),
    ).sort_values('n_cells', ascending=False)
    
    ct_summary.to_csv('ts_v2_cell_type_summary_FILTERED.csv')
    print(f"Number of cell types: {len(ct_summary)}")
    
    print(f"\n=== All cell types (top 50) ===")
    for ct, row in ct_summary.head(50).iterrows():
        print(f"  {ct:55s} | {row['n_cells']:7d} | {row['n_donors']:2d} donors | {row['n_tissues']:2d} tissues")
    
    # tissue summary
    tissue_summary = obs_df.groupby('tissue').agg(
        n_cells=('tissue', 'size'),
        n_cell_types=('cell_type', 'nunique'),
    ).sort_values('n_cells', ascending=False)
    
    tissue_summary.to_csv('ts_v2_tissue_summary_FILTERED.csv')
    print(f"\n=== Tissue summary ===")
    for tissue, row in tissue_summary.iterrows():
        print(f"  {tissue:35s} | {row['n_cells']:7d} cells | {row['n_cell_types']:3d} types")
    
    # pathway check
    print(f"\n{'='*70}")
    print("Step 4: pathway feasibility check")
    print(f"{'='*70}")
    
    pathways = [
        ("Fibroblast → Cardiomyocyte", ["fibroblast"], ["cardiomyocyte"]),
        ("Fibroblast → Hepatocyte", ["fibroblast"], ["hepatocyte"]),
        ("B cell → Macrophage", ["b cell", "naive b cell", "memory b cell"], ["macrophage"]),
        ("Fibroblast → Skeletal muscle", ["fibroblast"], ["skeletal muscle satellite cell", "slow muscle cell", "fast muscle cell"]),
        ("Fibroblast → Neuron (generic)", ["fibroblast"], ["neuron"]),
        ("Astrocyte → Neuron (generic)", ["astrocyte"], ["neuron"]),
        ("Pericyte → Neuron (generic)", ["pericyte"], ["neuron"]),
        ("Microglia → Neuron (generic)", ["microglial cell"], ["neuron"]),
        ("Fibroblast → Dopaminergic neuron", ["fibroblast"], ["dopaminergic neuron"]),
        ("Fibroblast → Motor neuron", ["fibroblast"], ["motor neuron"]),
        ("Fibroblast → GABAergic neuron", ["fibroblast"], ["gabaergic neuron"]),
        ("Astrocyte → Dopaminergic neuron", ["astrocyte"], ["dopaminergic neuron"]),
        ("Astrocyte → GABAergic neuron", ["astrocyte"], ["gabaergic neuron"]),
        ("Astrocyte → Glutamatergic neuron", ["astrocyte"], ["glutamatergic neuron"]),
        ("Hepatocyte → β cell", ["hepatocyte"], ["type b pancreatic cell", "pancreatic b cell"]),
        ("Hepatocyte → Neuron", ["hepatocyte"], ["neuron"]),
        ("Fibroblast → β cell", ["fibroblast"], ["type b pancreatic cell", "pancreatic b cell"]),
        ("Fibroblast → Osteoblast", ["fibroblast"], ["osteoblast"]),
        ("Fibroblast → Chondrocyte", ["fibroblast"], ["chondrocyte"]),
        ("Fibroblast → Adipocyte", ["fibroblast"], ["adipocyte"]),
        ("Fibroblast → Brown adipocyte", ["fibroblast"], ["brown preadipocyte", "brown fat cell"]),
        ("Fibroblast → Endothelial cell", ["fibroblast"], ["endothelial cell"]),
        ("Fibroblast → Smooth muscle cell", ["fibroblast"], ["smooth muscle cell", "vascular associated smooth muscle cell"]),
        ("Fibroblast → Keratinocyte", ["fibroblast"], ["keratinocyte"]),
        ("Fibroblast → Melanocyte", ["fibroblast"], ["melanocyte"]),
        ("Fibroblast → OPC", ["fibroblast"], ["oligodendrocyte precursor cell"]),
        ("Fibroblast → Schwann cell", ["fibroblast"], ["schwann cell"]),
        ("Fibroblast → Megakaryocyte", ["fibroblast"], ["megakaryocyte"]),
        ("Fibroblast → Erythroid progenitor", ["fibroblast"], ["erythroid progenitor cell", "erythroid lineage cell"]),
        ("Fibroblast → Hematopoietic progenitor", ["fibroblast"], ["hematopoietic stem cell"]),
        ("Fibroblast → Dendritic cell", ["fibroblast"], ["dendritic cell", "conventional dendritic cell"]),
        ("T cell → Macrophage", ["t cell", "cd4-positive, alpha-beta t cell", "cd8-positive, alpha-beta t cell"], ["macrophage"]),
        ("Fibroblast → Type II pneumocyte", ["fibroblast"], ["type ii pneumocyte", "lung epithelial cell", "alveolar type 2 fibroblast cell"]),
        ("Fibroblast → Intestinal epithelial", ["fibroblast"], ["enterocyte", "intestinal epithelial cell"]),
        ("Fibroblast → Renal tubular epithelial", ["fibroblast"], ["kidney epithelial cell", "epithelial cell of proximal tubule"]),
        ("Pancreatic exocrine → β cell", ["pancreatic acinar cell"], ["type b pancreatic cell", "pancreatic b cell"]),
        ("Glial cell (NG2) → Neuron", ["oligodendrocyte precursor cell"], ["neuron"]),
        ("T cell → β cell", ["t cell", "cd4-positive, alpha-beta t cell"], ["type b pancreatic cell", "pancreatic b cell"]),
    ]
    
    ct_lower = {ct.lower(): ct for ct in ct_summary.index}
    
    results = []
    for pw_name, source_queries, target_queries in pathways:
        src_cells = 0
        src_matches = []
        for sq in source_queries:
            for ct_l, ct_orig in ct_lower.items():
                if sq in ct_l:
                    n = ct_summary.loc[ct_orig, 'n_cells']
                    src_cells += n
                    src_matches.append(f"{ct_orig}({n})")
        
        tgt_cells = 0
        tgt_matches = []
        for tq in target_queries:
            for ct_l, ct_orig in ct_lower.items():
                if tq in ct_l:
                    n = ct_summary.loc[ct_orig, 'n_cells']
                    tgt_cells += n
                    tgt_matches.append(f"{ct_orig}({n})")
        
        feasible = src_cells >= 100 and tgt_cells >= 100
        
        results.append({
            'Pathway': pw_name,
            'Source_cells': src_cells,
            'Target_cells': tgt_cells,
            'Source_matches': '; '.join(src_matches) if src_matches else 'NOT FOUND',
            'Target_matches': '; '.join(tgt_matches) if tgt_matches else 'NOT FOUND',
            'Feasible': feasible,
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('ts_v2_pathway_feasibility_FILTERED.csv', index=False)
    
    feasible = df_results[df_results['Feasible']]
    not_feasible = df_results[~df_results['Feasible']]
    
    print(f"\nFeasible: {len(feasible)} / {len(df_results)}")
    
    print(f"\n--- ✓ feasible ---")
    for _, r in feasible.iterrows():
        print(f"  ✓ {r['Pathway']:42s} | src:{r['Source_cells']:7d} | tgt:{r['Target_cells']:7d}")
    
    print(f"\n--- ✗ not feasible ---")
    for _, r in not_feasible.iterrows():
        print(f"  ✗ {r['Pathway']:42s} | src:{r['Source_cells']:7d} | tgt:{r['Target_cells']:7d}")
        print(f"      Source: {r['Source_matches']}")
        print(f"      Target: {r['Target_matches']}")
    
    # save
    print(f"\nOutput files:")
    print(f"  ts_v2_cell_type_summary_FILTERED.csv")
    print(f"  ts_v2_tissue_summary_FILTERED.csv")
    print(f"  ts_v2_pathway_feasibility_FILTERED.csv")
