"""Embedding utilities for domain-specific models."""

import numpy as np
import torch


def load_student_projects(filepath):
    """
    Load student projects from CSV with custom separators.

    Args:
        filepath: Path to the student_project.csv file

    Returns:
        dict: Dictionary mapping author names to project content
    """
    projects = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip header line
    lines = content.strip().split('\n')
    if lines and '|' in lines[0] and 'author' in lines[0].lower():
        lines = lines[1:]

    # Rejoin and split by ###
    full_content = '\n'.join(lines)
    entries = full_content.split('###')

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Find the first | to separate author from content
        if '|' in entry:
            parts = entry.split('|', 1)
            author = parts[0].strip()
            content_text = parts[1].strip() if len(parts) > 1 else ""
            if author:
                projects[author] = content_text

    return projects


def get_text_embeddings(texts, model):
    """
    Get embeddings for a list of texts using a sentence transformer model.

    Args:
        texts: List of text strings
        model: SentenceTransformer model

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)


def get_transformer_embeddings(sequences, tokenizer, model, max_length=512, pooling='cls'):
    """
    Get embeddings from a HuggingFace transformer model.

    Args:
        sequences: List of sequences (SMILES, protein sequences, DNA, etc.)
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_length: Maximum sequence length
        pooling: 'cls' for [CLS] token, 'mean' for mean pooling

    Returns:
        numpy array of shape (n_sequences, embedding_dim)
    """
    embeddings = []

    model.eval()
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(
                seq,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if pooling == 'cls':
                # Use [CLS] token embedding (first token)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                # Mean pooling over sequence length
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings.append(emb[0])

    return np.array(embeddings)


def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix for embeddings.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
        numpy array of shape (n_samples, n_samples)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def find_top_similar_pairs(similarity_matrix, labels, top_k=5):
    """
    Find the top-k most similar pairs from a similarity matrix.

    Args:
        similarity_matrix: Square similarity matrix
        labels: List of labels for each sample
        top_k: Number of top pairs to return

    Returns:
        List of tuples (similarity, label1, label2)
    """
    pairs = []
    n = len(labels)

    for i in range(n):
        for j in range(i+1, n):
            pairs.append((similarity_matrix[i, j], labels[i], labels[j]))

    pairs.sort(reverse=True)
    return pairs[:top_k]


def find_most_similar(query_embedding, corpus_embeddings, corpus_labels, top_k=5):
    """
    Find the most similar items to a query.

    Args:
        query_embedding: Query embedding (1D array)
        corpus_embeddings: Corpus embeddings (2D array)
        corpus_labels: Labels for corpus items
        top_k: Number of results to return

    Returns:
        List of tuples (similarity, label)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    results = [(similarities[i], corpus_labels[i]) for i in range(len(corpus_labels))]
    results.sort(reverse=True)
    return results[:top_k]
