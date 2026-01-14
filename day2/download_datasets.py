"""
Download and sample datasets for TP2: Embeddings
Run: python download_datasets.py
"""

import os
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "day2/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_molecules():
    """Download BBBP molecules dataset (full)."""
    print("\n=== Downloading Molecules (BBBP) ===")

    ds = load_dataset('scikit-fingerprints/MoleculeNet_BBBP', split='train')
    df = pd.DataFrame(ds)

    # Use all data
    df = df[['SMILES', 'label']].copy()
    df['label_name'] = df['label'].map({1: 'BBB_permeable', 0: 'BBB_impermeable'})

    out_path = f"{OUTPUT_DIR}/molecules_bbbp.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} molecules to {out_path}")
    print(f"  Labels: {df['label_name'].value_counts().to_dict()}")

    return df


def download_proteins():
    """Download proteins from selected Pfam families (full)."""
    print("\n=== Downloading Proteins (SwissProt-Pfam) ===")

    # Famous protein families with distinct functions
    target_families = {
        'PF00118': 'TCP1_chaperonin',      # Chaperonin
        'PF00005': 'ABC_transporter',      # ABC transporter
        'PF00001': 'GPCR_rhodopsin',       # G-protein coupled receptor
        'PF00002': 'GPCR_secretin',        # Secretin family
        'PF00003': 'GPCR_metabotropic',    # Metabotropic glutamate
    }

    print("  Loading SwissProt-Pfam dataset...")
    ds = load_dataset('DanielHesslow/SwissProt-Pfam', split='train')
    df = pd.DataFrame(ds)

    print(f"  Total proteins: {len(df)}")

    # Get all proteins from target families
    sampled = []
    for pfam_id, family_name in target_families.items():
        mask = df['labels_str'].str.contains(pfam_id, na=False)
        family_df = df[mask].copy()

        if len(family_df) > 0:
            family_df['family_id'] = pfam_id
            family_df['family_name'] = family_name
            sampled.append(family_df)
            print(f"  {pfam_id} ({family_name}): {len(family_df)} sequences")
        else:
            print(f"  {pfam_id} ({family_name}): NOT FOUND")

    if sampled:
        df_out = pd.concat(sampled, ignore_index=True)
        df_out = df_out.rename(columns={'seq': 'sequence', 'id': 'uniprot_id'})
        df_out = df_out[['uniprot_id', 'sequence', 'family_id', 'family_name']]

        out_path = f"{OUTPUT_DIR}/proteins_pfam.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  Saved {len(df_out)} proteins to {out_path}")
    else:
        print("  ERROR: No proteins found for target families")
        df_out = pd.DataFrame()

    return df_out


def download_dna():
    """Download DNA sequences - histone modification prediction (full H3 task)."""
    print("\n=== Downloading DNA (Nucleotide Transformer) ===")

    ds = load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks', split='train')
    df = pd.DataFrame(ds)

    print(f"  Total sequences: {len(df)}")

    # Use H3 histone modification task (full)
    df_out = df[df['task'] == 'H3'].copy()
    df_out['label_name'] = df_out['label'].map({1: 'H3_modified', 0: 'H3_unmodified'})
    df_out = df_out[['sequence', 'label', 'label_name']]

    out_path = f"{OUTPUT_DIR}/dna_histone.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved {len(df_out)} sequences to {out_path}")
    print(f"  Sequence length: {len(df_out['sequence'].iloc[0])} bp")
    print(f"  Labels: {df_out['label_name'].value_counts().to_dict()}")

    return df_out


if __name__ == "__main__":
    print("Downloading datasets for TP2...")

    molecules = download_molecules()
    proteins = download_proteins()
    dna = download_dna()

    print("\n=== Summary ===")
    print(f"  Molecules: {len(molecules)} samples")
    print(f"  Proteins: {len(proteins)} samples")
    print(f"  DNA: {len(dna)} samples")
    print(f"\nFiles saved to {OUTPUT_DIR}/")
