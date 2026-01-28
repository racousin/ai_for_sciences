# AI for Sciences - Winter School Practical Works

This repository contains the practical work materials for the [Joint Winter School on AI for Sciences](https://event.ntu.edu.sg/Joint-Winter-School-AI-for-Sciences).

## Getting Started

All notebooks can be executed directly in **Google Colab** (Gmail account required). Click the "Open in Colab" badge at the top of each notebook.

Alternatively, install dependencies locally:
```bash
pip install git+https://github.com/racousin/ai_for_sciences.git
```

## Schedule

### Day 1 - Foundations

| Notebook | Topic | Colab Link |
|----------|-------|------------|
| `day1/tp-ml-phd.ipynb` | Machine Learning: Movie Recommendation System | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day1/tp-ml-phd.ipynb) |
| `day1/tp2_bonus.ipynb` | **Bonus:** Optimization & Neural Networks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day1/tp2_bonus.ipynb) |

**Learning objectives:**
- Exploratory data analysis and preprocessing
- Regression and classification with scikit-learn
- Model validation and hyperparameter tuning
- Gradient descent optimization with PyTorch
- CPU vs GPU performance comparison

---

### Day 2 - Embeddings & LLMs

| Notebook | Topic | Colab Link |
|----------|-------|------------|
| `day2/tp1_part1.ipynb` | Tokenization & Embeddings Theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day2/tp1_part1.ipynb) |
| `day2/tp1_part2.ipynb` | Domain-Specific Embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day2/tp1_part2.ipynb) |
| `day2/tp1_part3.ipynb` | Classification with Embeddings (BBB Prediction) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day2/tp1_part3.ipynb) |
| `day2/tp2_bonus.ipynb` | **Bonus:** Context Engineering & LoRA Fine-tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day2/tp2_bonus.ipynb) |

**Learning objectives:**
- Understand the Text → Tokens → Embeddings pipeline
- Use domain-specific pre-trained models (SciBERT, ChemBERTa, ESM-2, Nucleotide Transformer)
- Apply transfer learning with frozen embeddings
- Master prompting strategies: zero-shot, few-shot, chain-of-thought
- Fine-tune models efficiently with LoRA

**Datasets:**
- `molecules_bbbp.csv` - 2,039 molecules with blood-brain barrier labels
- `proteins_pfam.csv` - 5,789 proteins from 5 Pfam families
- `dna_histone.csv` - 13,468 DNA sequences (histone modification)
- `maths.csv` - 900 math problems (arithmetic, algebra, geometry, etc.)

---

### Day 3 - Computer Vision & Generative Models

| Notebook | Topic | Colab Link |
|----------|-------|------------|
| `day3/tp1.ipynb` | Transfer Learning for Medical Image Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day3/tp1.ipynb) |
| `day3/tp1_solutions.ipynb` | Solutions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day3/tp1_solutions.ipynb) |
| `day3/tp2.ipynb` | Object Detection with YOLO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day3/tp2.ipynb) |
| `day3/tp3_bonus.ipynb` | **Bonus:** Face Generation with DCGAN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_sciences/blob/main/day3/tp3_bonus.ipynb) |

**Learning objectives:**
- Data augmentation for medical images
- Transfer learning: frozen backbone vs full fine-tuning
- Object detection with YOLO and bounding box annotations
- Fine-tuning YOLO on custom datasets
- Understanding GANs (and Exploring latent space interpolation)

---

## Repository Structure

```
ai_for_sciences/
├── aiforscience/           # Utility package (pip installable)
│   ├── visualization.py    # Plotting functions
│   ├── display.py          # Print/formatting utilities
│   └── data.py             # Data generation utilities
├── day1/
│   ├── tp-ml-phd.ipynb           # ML fundamentals (recommendation)
│   ├── tp-ml-phd-solutions.ipynb # Solutions
│   └── tp2_bonus.ipynb           # Optimization & PyTorch
├── day2/
│   ├── tp1_part1.ipynb     # Tokenization & Embeddings
│   ├── tp1_part2.ipynb     # Domain-specific embeddings
│   ├── tp1_part3.ipynb     # Classification with embeddings
│   ├── tp2_bonus.ipynb     # Context engineering & LoRA
│   └── data/               # Datasets
├── day3/
│   ├── tp1.ipynb           # Transfer learning (BloodMNIST)
│   ├── tp1_solutions.ipynb # Solutions
│   ├── tp2.ipynb           # Object detection (YOLO)
│   └── tp3_bonus.ipynb     # Generative models (DCGAN)
└── pyproject.toml
```

## Models Used

| Domain | Model | Notebook |
|--------|-------|----------|
| Text | `allenai/scibert_scivocab_uncased` | day2/tp1_part2 |
| Text | `sentence-transformers/all-MiniLM-L6-v2` | day2/tp1_part1 |
| Molecules | `seyonec/ChemBERTa-zinc-base-v1` | day2/tp1_part2, tp1_part3 |
| Proteins | `facebook/esm2_t6_8M_UR50D` | day2/tp1_part2 |
| DNA | `InstaDeepAI/nucleotide-transformer-500m-human-ref` | day2/tp1_part2 |
| Language | `Qwen/Qwen2-0.5B-Instruct` | day2/tp2_bonus |
| Vision | `ResNet18` (torchvision) | day3/tp1 |
| Detection | `YOLOv8n` (ultralytics) | day3/tp2 |

## License

MIT
