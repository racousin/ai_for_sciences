# TP2: Representing Scientific Data (Tokenization + Embeddings)

## Goal
One unified TP showing how different scientific domains represent data → tokenize → embed → visualize similarity.

## What We Have
- `student_project.csv`: 15 student project descriptions (in progress)

## Datasets Status

| Domain | File | Samples | Labels |
|--------|------|---------|--------|
| **Text** | `student_project.csv` | ⚠️ 3/15 | Field/topic |
| **Molecules** | `data/molecules_bbbp.csv` | ✅ 2,039 | 1,560 permeable / 479 impermeable |
| **Proteins** | `data/proteins_pfam.csv` | ✅ 5,789 | 5 Pfam families |
| **DNA** | `data/dna_histone.csv` | ✅ 13,468 | 6,900 modified / 6,568 unmodified |

### Downloaded Data Details

**Molecules** (Blood-Brain Barrier Penetration)
- Source: `scikit-fingerprints/MoleculeNet_BBBP`
- Columns: `SMILES`, `label`, `label_name`

**Proteins** (5 distinct Pfam families)
- Source: `DanielHesslow/SwissProt-Pfam`
- Families:
  - PF00005 (ABC_transporter): 3,173
  - PF00001 (GPCR_rhodopsin): 1,404
  - PF00118 (TCP1_chaperonin): 976
  - PF00002 (GPCR_secretin): 152
  - PF00003 (GPCR_metabotropic): 84
- Columns: `uniprot_id`, `sequence`, `family_id`, `family_name`

**DNA** (Histone H3 modification)
- Source: `InstaDeepAI/nucleotide_transformer_downstream_tasks`
- Task: H3 histone modification prediction
- Columns: `sequence`, `label`, `label_name`
- Sequence length: 500 bp

### Regenerate Data
```bash
poetry run python day2/download_datasets.py
```

## Models to Use (HuggingFace)

| Domain | Model 1 | Model 2 (alternative) |
|--------|---------|----------------------|
| **Text** | `allenai/scibert_scivocab_uncased` | `sentence-transformers/all-MiniLM-L6-v2` | 
| **Molecules** | `seyonec/ChemBERTa-zinc-base-v1` | `DeepChem/ChemBERTa-77M-MTR` |
| **Proteins** | `facebook/esm2_t6_8M_UR50D` | `Rostlab/prot_bert` |
| **DNA** | `zhihan1996/DNABERT-2-117M` | `InstaDeepAI/nucleotide-transformer-500m-human-ref` |

## Structure per Domain
1. **Data**: Show raw format (SMILES, amino acids, nucleotides)
2. **Tokenization**: How it's tokenized (subword, k-mer, per-residue)
3. **Embedding**: Extract embeddings with model
4. **Visualization**: t-SNE/PCA
5. **Similarity**: Matrix heatmap + nearest neighbors

## TODO
- [x] Download molecule dataset → `data/molecules_bbbp.csv` (200 samples)
- [x] Download protein dataset → `data/proteins_pfam.csv` (100 samples, 5 families)
- [x] Download DNA dataset → `data/dna_histone.csv` (100 samples)
- [x] Create download script → `download_datasets.py`
- [ ] **Complete student_project.csv (need 12+ more entries)**
- [ ] Test each HuggingFace model loads in Colab
- [x] Create utility functions in `aiforscience/`:
  - `plot_similarity_matrix()`
  - `plot_embeddings_tsne()`
- [ ] Build notebook Part 1 (exploration)
- [ ] Decide on Part 2 (classification) later

## Student Projects CSV Status
- **Format**: `author | content` with `###` as record separator
- **Current entries**: 3
- **Need**: 12-22 more to reach 15-25 total
- **Fields covered**: GIS/mapping, EEG/brain, scRNA-seq/biology
