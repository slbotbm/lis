import json
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from pydantic import BaseModel, ValidationError, validator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# --- Pydantic Model for Validation ---
class ModelResponse(BaseModel):
    """
    Pydantic class to validate the structure of the JSON response.
    It ensures that 'model_response' is a list of lists, where each inner list
    contains exactly three strings.
    """

    model_response: List[List[str]]

    @validator("model_response")
    def check_triplet_length(cls, v):
        """
        Validates that each inner list in model_response contains exactly 3 strings.
        """
        for i, sublist in enumerate(v):
            if len(sublist) != 3:
                raise ValueError(
                    f"Triplet at index {i} does not contain 3 elements. Found {len(sublist)}."
                )
            if not all(isinstance(item, str) for item in sublist):
                raise TypeError(f"All items in triplet at index {i} must be strings.")
        return v


def process_directory(
    target_path: Path, similarity_thresholds: List[float], model: SentenceTransformer
):
    """
    Processes a single directory (e.g., 1b_3b, 4b_6b, 7b_9b), validates JSONs,
    creates embeddings, and generates clusters.

    Args:
        target_path (Path): The path to the directory to process.
        similarity_thresholds (List[float]): A list of cosine similarity percentages for clustering.
        model (SentenceTransformer): The pre-loaded sentence transformer model.
    """
    logger.info(f"Processing directory: {target_path}")
    all_valid_triplets = []

    json_files = list(target_path.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {target_path}. Skipping.")
        return

    # --- Validation and Data Aggregation ---
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate the data using the Pydantic model
            ModelResponse(**data)
            data["output_format_correct"] = True
            all_valid_triplets.extend(data["model_response"])

        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {json_file}.")
            # We can't add output_format_correct if we can't load it
            continue
        except ValidationError as e:
            logger.warning(f"Validation failed for {json_file}: {e}")
            data["output_format_correct"] = False
        except Exception as e:
            logger.error(f"An unexpected error occurred with {json_file}: {e}")
            data["output_format_correct"] = False

        # --- Update JSON file with validation status ---
        # This part adds the 'output_format_correct' key back to the original file.
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    if not all_valid_triplets:
        logger.warning(
            f"No valid triplets found in {target_path}. Skipping clustering."
        )
        return

    # --- Sentence Conversion and Embedding ---
    logger.info(f"Found {len(all_valid_triplets)} valid triplets. Creating embeddings.")
    sentences = [" ".join(triplet) for triplet in all_valid_triplets]
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Create a mapping from sentence to vector
    embedding_dict = {
        sentence: vector.tolist() for sentence, vector in zip(sentences, embeddings)
    }

    # --- Cosine Similarity Matrix ---
    # Using numpy for efficient calculation
    embeddings_normalized = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    cosine_similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

    # --- Clustering for each percentage ---
    for percentage in similarity_thresholds:
        logger.info(f"Clustering with {percentage * 100}% similarity threshold...")

        # Convert similarity to distance for the clustering algorithm
        distance_threshold = 1 - percentage

        # Using Agglomerative Clustering
        distance_matrix = 1 - cosine_similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        ).fit(distance_matrix)

        labels = clustering.labels_

        # Organize triplets into clusters
        clusters = {}
        for i, label in enumerate(labels):
            label_str = str(label)  # Use string keys for JSON compatibility
            if label_str not in clusters:
                clusters[label_str] = []
            clusters[label_str].append(all_valid_triplets[i])

        # Convert clusters dictionary to a list of lists for the final JSON output
        cluster_list = list(clusters.values())

        # --- Save Clusters to JSON ---
        output_filename = target_path / f"cluster_{percentage}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(cluster_list, f, indent=4)
        logger.success(f"Saved {len(cluster_list)} clusters to {output_filename}")


def main():
    """
    Main function to traverse directories and initiate processing.
    """
    # Define the percentages for clustering as required
    CLUSTER_PERCENTAGES = [0.99, 0.95, 0.90]

    # --- Setup Paths ---
    # The script is in src/evals/, so we need to go up two directories to the root.
    root_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = root_dir / "results"

    if not results_dir.exists():
        logger.error(f"Results directory not found at: {results_dir}")
        return

    # --- Load Model ---
    # This model is good for semantic similarity tasks.
    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded.")

    # --- Directory Traversal ---
    numbered_folders = [
        d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]

    for num_folder in sorted(numbered_folders, key=lambda p: int(p.name)):
        for sub_folder_name in ["1b_3b", "4b_6b", "7b_9b"]:
            target_path = num_folder / sub_folder_name
            if target_path.exists() and target_path.is_dir():
                process_directory(target_path, CLUSTER_PERCENTAGES, model)
            else:
                logger.warning(f"Subdirectory {target_path} not found. Skipping.")


if __name__ == "__main__":
    # Configure Loguru to write to a log file
    log_file_path = Path(__file__).parent / "processing_{time}.log"
    logger.add(log_file_path)
    main()
