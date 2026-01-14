# TP2: Representing Scientific Data (Tokenization + Embeddings)

## Goal
One unified TP showing how different scientific domains represent data → tokenize → embed → visualize similarity.

## What We Have
- `student_project.csv`: 15 student project descriptions (in progress)

## Datasets Needed

| Domain | Source | Target Size | Labels |
|--------|--------|-------------|--------|
| **Text** | student_project.csv | 15-25 | Field/topic |
| **Molecules** | MoleculeNet (BBBP, Tox21) or ChEMBL | ~100-200 | Drug class, activity |
| **Proteins** | UniProt subset | ~100 | Protein family |
| **DNA** | EPD / ENCODE / HuggingFace | ~100 | Promoter type, function |

### Dataset URLs to check:
- MoleculeNet: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/`
- UniProt REST: `https://rest.uniprot.org/`
- HuggingFace datasets: `datasets` library

## Models to Use (HuggingFace)

| Domain | Model 1 | Model 2 (alternative) |
|--------|---------|----------------------|
| **Text** | `sentence-transformers/all-MiniLM-L6-v2` | `allenai/scibert_scivocab_uncased` |
| **Molecules** | `seyonec/ChemBERTa-zinc-base-v1` | `DeepChem/ChemBERTa-77M-MTR` |
| **Proteins** | `facebook/esm2_t6_8M_UR50D` | `Rostlab/prot_bert` |
| **DNA** | `zhihan1996/DNABERT-2-117M` | `InstaDeepAI/nucleotide-transformer-500m-human-ref` |

## Structure per Domain
1. **Data**: Show raw format (SMILES, amino acids, nucleotides)
2. **Tokenization**: How it's tokenized (subword, k-mer, per-residue)
3. **Embedding**: Extract embeddings with model
4. **Visualization**: UMAP/t-SNE/PCA
5. **Similarity**: Matrix heatmap + nearest neighbors

## TODO
- [ ] Download/sample molecule dataset (~200 samples with categories)
- [ ] Download/sample protein dataset (~100 sequences with families)
- [ ] Download/sample DNA dataset (~100 sequences with labels)
- [ ] Test each HuggingFace model loads in Colab
- [ ] Create utility functions in `aiforscience/`:
  - `plot_similarity_matrix()`
  - `plot_embeddings_umap()`
  - `load_molecule_dataset()`, `load_protein_dataset()`, `load_dna_dataset()`
- [ ] Build notebook Part 1 (exploration)
- [ ] Decide on Part 2 (classification) later
