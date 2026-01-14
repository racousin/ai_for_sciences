# AI for Sciences - Winter School Practical Works

This repository contains the practical work materials for the [Joint Winter School on AI for Sciences](https://event.ntu.edu.sg/Joint-Winter-School-AI-for-Sciences).

## Target Audience

25 international PhD students (2-4 years into their PhD) from various scientific fields involving ML (biology, chemistry, physics, medicine, earth sciences, etc.)

## Getting Started

All notebooks can be executed directly in **Google Colab** (Gmail account required). Click the "Open in Colab" badge at the top of each notebook.

Alternatively, install dependencies locally:
```bash
pip install git+https://github.com/racousin/ai_for_sciences.git
```

## Schedule

### Day 1 (2pm - 5pm) - Foundations

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `day1/tp1.ipynb` | Optimization & Neural Networks | Gradient descent, linear regression, PyTorch basics, CPU vs GPU |

**Learning objectives:**
- Understand gradient descent optimization
- Build and train models with PyTorch (`nn.Linear`, `nn.MSELoss`, `torch.optim.SGD`)
- Compare CPU vs GPU performance

---

### Day 2 (2pm - 5pm) - Embeddings & LLMs

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `day2/tp1_part1.ipynb` | Tokenization & Embeddings Theory | Text → Tokens → Embeddings pipeline, cosine similarity, pre-trained models |
| `day2/tp1_part2.ipynb` | Domain-Specific Embeddings | Scientific text, molecules (ChemBERTa), proteins (ESM-2), DNA (DNABERT-2) |
| `day2/tp1_part3.ipynb` | Classification with Embeddings | Transfer learning, MLP classifier, BBB permeability prediction |
| `day2/tp2.ipynb` | Context Engineering & Fine-tuning | Prompting strategies, LoRA fine-tuning, math problem solving |

**Learning objectives:**
- Understand embeddings as dense vector representations
- Use domain-specific pre-trained models (molecules, proteins, DNA)
- Apply transfer learning with frozen embeddings
- Master prompting strategies: zero-shot, few-shot, chain-of-thought
- Fine-tune models efficiently with LoRA

**Datasets:**
- `molecules_bbbp.csv` - 2,039 molecules with blood-brain barrier labels
- `proteins_pfam.csv` - 5,789 proteins from 5 Pfam families
- `dna_histone.csv` - 13,468 DNA sequences (histone modification)
- `maths.csv` - 900 math problems (arithmetic, algebra, geometry, etc.)

---

### Day 3 (2pm - 5pm) - Computer Vision & Generative Models

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| `day3/tp1.ipynb` | Object Detection with YOLO | YOLO architecture, fine-tuning on scientific imagery |
| `day3/tp2.ipynb` | Generative Models | GAN basics |

---

## Repository Structure

```
ai_for_sciences/
├── aiforscience/           # Utility package (pip installable)
│   ├── visualization.py    # Plotting functions
│   ├── display.py          # Print/formatting utilities
│   └── data.py             # Data generation utilities
├── day1/
│   └── tp1.ipynb
├── day2/
│   ├── tp1_part1.ipynb
│   ├── tp1_part2.ipynb
│   ├── tp1_part3.ipynb
│   ├── tp2.ipynb
│   └── data/
├── day3/
└── pyproject.toml
```

## Models Used

| Domain | Model | Notebook |
|--------|-------|----------|
| Text | `sentence-transformers/all-MiniLM-L6-v2` | tp1_part1, tp1_part2 |
| Molecules | `seyonec/ChemBERTa-zinc-base-v1` | tp1_part2 |
| Proteins | `facebook/esm2_t6_8M_UR50D` | tp1_part2 |
| DNA | `zhihan1996/DNABERT-2-117M` | tp1_part2 |
| Language | `gpt2` | tp2 |

## License

MIT
