"""AI for Sciences - Utility Package for Practicals"""

from .visualization import (
    plot_gradient_descent_1d,
    plot_loss_history,
    plot_predictions,
    plot_gradient_step,
    plot_nonlinear_data,
    plot_attention_weights,
    plot_embeddings_2d,
    visualize_tokens,
    compare_tokenizers,
    tokenizer_stats,
    plot_similarity_matrix,
    plot_embeddings_umap,
    plot_embeddings_tsne,
    print_tokenization_example,
)

from .display import (
    print_model_params,
    print_device_comparison,
    print_tokenization,
    print_model_summary,
)

from .data import (
    generate_linear_data,
    generate_nonlinear_data,
)

from .embeddings import (
    load_student_projects,
    get_text_embeddings,
    get_transformer_embeddings,
    compute_similarity_matrix,
    find_top_similar_pairs,
    find_most_similar,
    semantic_search,
    print_search_results,
)

__all__ = [
    # Visualization
    "plot_gradient_descent_1d",
    "plot_loss_history",
    "plot_predictions",
    "plot_gradient_step",
    "plot_nonlinear_data",
    "plot_attention_weights",
    "plot_embeddings_2d",
    "visualize_tokens",
    "compare_tokenizers",
    "tokenizer_stats",
    "plot_similarity_matrix",
    "plot_embeddings_umap",
    "plot_embeddings_tsne",
    "print_tokenization_example",
    # Display
    "print_model_params",
    "print_device_comparison",
    "print_tokenization",
    "print_model_summary",
    # Data
    "generate_linear_data",
    "generate_nonlinear_data",
    # Embeddings
    "load_student_projects",
    "get_text_embeddings",
    "get_transformer_embeddings",
    "compute_similarity_matrix",
    "find_top_similar_pairs",
    "find_most_similar",
    "semantic_search",
    "print_search_results",
]
