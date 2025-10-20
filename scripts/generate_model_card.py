import yaml
import re
import argparse

def parse_config(config_path):
    """Parses the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_results(results_path):
    """Parses the C2ST results file."""
    if not results_path:
        return {}
    try:
        with open(results_path, 'r') as f:
            content = f.read()
        match = re.search(r"Average C2ST accuracy: ([\d.]+) \+- ([\d.]+)", content)
        if match:
            return {"mean_accuracy": float(match.group(1)), "std_dev": float(match.group(2))}
    except FileNotFoundError:
        print(f"Warning: Results file not found at {results_path}")
    return {}

def create_markdown_content(config, results):
    """Generates the simple Markdown content for the model card."""

    # --- Extract basic info ---
    model_name = config.get('strategy', {}).get('model', 'N/A')
    method = config.get('strategy', {}).get('method', 'N/A')
    task_name = config.get('task_name', 'N/A')
    pipeline_type = "Flow Matching" if method == "flow" else "Diffusion"

    # --- Helper function to create Markdown tables ---
    def create_table_from_dict(data_dict):
        header = "| Parameter | Value |\n|---|---|"
        rows = [f"| `{key}` | `{value}` |" for key, value in data_dict.items()]
        return header + "\n" + "\n".join(rows)

    # --- Create tables for model architecture and training config ---
    model_arch_table = create_table_from_dict(config.get('model', {}))
    
    training_params = config.get('training', {})
    optimizer_params = config.get('optimizer', {})
    full_training_config = {**training_params, **optimizer_params}
    training_config_table = create_table_from_dict(full_training_config)

    # --- Build the final Markdown string ---
    md_content = f"""
# Model Card: {model_name.capitalize()} on {task_name}

This document provides a summary of the `{model_name}` model trained on the `{task_name}` dataset.

## 1. Model & Pipeline

- **Model Architecture:** `{model_name}`
- **Training Pipeline:** `{pipeline_type}`
- **Purpose:** Reconstruct posterior distributions in a Simulation-Based Inference (SBI) context.

## 2. Dataset

- **Dataset:** `{task_name}`
- **Description:** A synthetic benchmark dataset.
- **Training Size:** The model was trained on 100,000 (1e5) samples.

## 3. Model Architecture

{model_arch_table}

## 4. Training Configuration

{training_config_table}

## 5. Evaluation

The model's performance is evaluated using the Classifier 2-Sample Test (C2ST). An accuracy score close to 0.5 indicates that the generated samples are highly similar to the true data distribution.

- **Average C2ST Accuracy:** {results.get('mean_accuracy', 'N/A'):.3f} ± {results.get('std_dev', 'N/A'):.3f}

---
*This model card was automatically generated.*
"""
    return md_content

def main():
    parser = argparse.ArgumentParser(description="Generate a simple Markdown model card.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--c2st", type=str, required=True, help="Path to the C2ST results text file.")
    parser.add_argument("-o", "--output", type=str, default="README.md", help="Path to save the output model card.")
    args = parser.parse_args()

    config_data = parse_config(args.config)
    results_data = parse_results(args.c2st)

    markdown = create_markdown_content(config_data, results_data)

    with open(args.output, 'w') as f:
        f.write(markdown)

    print(f"Successfully created simple model card at {args.output}!")

if __name__ == "__main__":
    main()

