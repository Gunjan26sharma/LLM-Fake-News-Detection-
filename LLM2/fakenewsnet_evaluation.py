import os
import pandas as pd
import numpy as np
import time
import json
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()


# Constants
class Config:
    # Paths
    DATA_DIR = "data/"
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed/")
    RESULTS_DIR = "results/"
    FAKENEWSNET_RESULTS_DIR = os.path.join(RESULTS_DIR, "fakenewsnet/")

    # Dataset files
    TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
    TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")
    VALID_FILE = os.path.join(PROCESSED_DIR, "valid.csv")

    # Labels
    LABELS = ["fake", "real"]

    # Sources
    SOURCES = ["gossipcop", "politifact"]

    # Evaluation settings
    SAMPLE_SIZE = 200  # Set to None for full dataset evaluation

    # Model settings
    MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.0

    # Strategy settings - will be loaded from best_strategy.json if available
    PROMPT_STRATEGY = "detailed"  # Default if no best strategy found


# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")


def load_data(file_path):
    """Load and parse CSV files."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Preprocess the data."""
    processed_data = data.copy()

    # Ensure title is cleaned
    processed_data['title'] = processed_data['title'].fillna("")

    # Select relevant columns based on what's available
    # For FakeNewsNet, we focus on title, source, and label
    relevant_columns = ['id', 'label', 'title', 'source', 'domain', 'tweet_count']

    relevant_columns = [col for col in relevant_columns if col in processed_data.columns]

    return processed_data[relevant_columns]


def load_best_strategy():
    """Load the best prompt strategy if available."""
    strategy_file = os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "best_strategy.json")

    if os.path.exists(strategy_file):
        with open(strategy_file, 'r') as f:
            strategy_data = json.load(f)

        return strategy_data["strategy_name"]
    else:
        return Config.PROMPT_STRATEGY


# Define prompt strategies - same as in prompt_engineering.py
prompt_strategies = {
    "basic": {
        "system": "You are a fact-checker. Classify the news title as either 'real' or 'fake'. Reply with just one word.",
        "user": "News Title: \"{title}\"\nClassify as 'real' or 'fake':"
    },

    "detailed": {
        "system": """You are a highly accurate fake news detector. Your task is to classify news titles as either "real" or "fake".
- Classify as "fake" if the news title appears to be false, misleading, exaggerated, or clickbait.
- Classify as "real" if the news title appears to be factually accurate and journalistically sound.
Respond with only one word: either "real" or "fake".""",
        "user": """News Title: "{title}"
Source: {source}
Domain: {domain}

Classify the above news title as "real" or "fake"."""
    },

    "source_aware": {
        "system": """You are a highly accurate fake news detector specializing in {source} content.
Your task is to classify news titles as either "real" or "fake".
Respond with only one word: either "real" or "fake".""",
        "user": """News Title: "{title}"
Domain: {domain}

Based on your knowledge of {source} content, classify the above news title as "real" or "fake"."""
    },

    "efficient": {
        "system": "Classify: real or fake? One word only.",
        "user": "\"{title}\""
    },

    "few_shot": {
        "system": "You are a fact-checker. Classify news titles as 'real' or 'fake'. Follow the examples.",
        "user": """Examples:
Title: "BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs"
Classification: fake

Title: "Teen Mom Star Jenelle Evans' Wedding Dress Is Available Here for $2999"
Classification: real

Title: "Court Orders Obama To Pay $400 Million In Restitution"
Classification: fake

Title: "I Tried Kim Kardashian's Butt Workout & Am Forever Changed"
Classification: real

Title: "{title}"
Classification:"""
    }
}


def format_prompt(row, strategy_name):
    """Format a prompt based on the specified strategy."""
    strategy = prompt_strategies[strategy_name]

    system_template = strategy["system"]
    user_template = strategy["user"]

    # Replace {source} in system prompt if needed
    if "{source}" in system_template:
        system_template = system_template.format(source=row['source'])

    # Fill in the templates
    title = row['title']
    source = row['source'] if 'source' in row and not pd.isna(row['source']) else "unknown source"
    domain = row['domain'] if 'domain' in row and not pd.isna(row['domain']) else "unknown domain"

    user_prompt = user_template.format(
        title=title,
        source=source,
        domain=domain
    )

    return system_template, user_prompt


def classify_with_llm(system_prompt, user_prompt):
    """Send a prompt to the LLM API and return the result and metrics."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=Config.TEMPERATURE,
            max_tokens=10
        )

        prediction = response.choices[0].message.content.strip().lower()
        processing_time = time.time() - start_time
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return prediction, token_usage, processing_time

    except Exception as e:
        print(f"Error in API call: {e}")
        return None, None, None


def run_evaluation(data, strategy_name, sample_size=None):
    """Run a full evaluation using the specified strategy."""
    if sample_size and len(data) > sample_size:
        eval_data = data.sample(sample_size, random_state=42)
    else:
        eval_data = data

    results = []
    true_labels = []
    predicted_labels = []

    total_tokens = 0
    total_time = 0

    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Evaluating"):
        system_prompt, user_prompt = format_prompt(row, strategy_name)
        prediction, token_usage, processing_time = classify_with_llm(system_prompt, user_prompt)

        if prediction and token_usage:
            true_label = row['label']

            # Normalize prediction to match expected labels
            if prediction not in ["real", "fake"]:
                # Simple heuristic for non-exact matches
                prediction = "real" if any(t in prediction for t in ["real", "true", "accurate"]) else "fake"

            results.append({
                "id": row['id'],
                "title": row['title'],
                "true_label": true_label,
                "predicted_label": prediction,
                "tokens": token_usage,
                "processing_time": processing_time
            })

            true_labels.append(true_label)
            predicted_labels.append(prediction)

            total_tokens += token_usage["total_tokens"]
            total_time += processing_time

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, pos_label="real", average="binary"),
        "recall": recall_score(true_labels, predicted_labels, pos_label="real", average="binary"),
        "f1": f1_score(true_labels, predicted_labels, pos_label="real", average="binary"),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=["fake", "real"]).tolist()
    }

    # Calculate efficiency metrics
    efficiency = {
        "avg_tokens_per_title": total_tokens / len(results),
        "avg_processing_time": total_time / len(results),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "estimated_cost": (total_tokens / 1000) * 0.002  # Assuming $0.002 per 1K tokens for GPT-3.5-turbo
    }

    return results, metrics, efficiency


def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_errors(results_data):
    """Analyze error patterns in the results."""
    df = pd.DataFrame(results_data)

    # Filter to incorrect predictions
    errors = df[df['true_label'] != df['predicted_label']]

    # Group by true label
    error_by_true_label = errors.groupby('true_label').size().reset_index(name='count')

    # Group by predicted label
    error_by_pred_label = errors.groupby('predicted_label').size().reset_index(name='count')

    # Plot error distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x='true_label', y='count', data=error_by_true_label)
    plt.title('Errors by True Label')
    plt.xlabel('True Label')
    plt.ylabel('Error Count')

    plt.subplot(1, 2, 2)
    sns.barplot(x='predicted_label', y='count', data=error_by_pred_label)
    plt.title('Errors by Predicted Label')
    plt.xlabel('Predicted Label')
    plt.ylabel('Error Count')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "error_analysis.png"))
    plt.close()

    # Find examples of each error type (for binary classification)
    error_types = {}

    # False positives (fake classified as real)
    false_positives = errors[(errors['true_label'] == 'fake') & (errors['predicted_label'] == 'real')]
    error_types['false_positives'] = false_positives.head(5).to_dict('records')

    # False negatives (real classified as fake)
    false_negatives = errors[(errors['true_label'] == 'real') & (errors['predicted_label'] == 'fake')]
    error_types['false_negatives'] = false_negatives.head(5).to_dict('records')

    return {
        "error_counts": {
            "by_true_label": error_by_true_label.to_dict('records'),
            "by_predicted_label": error_by_pred_label.to_dict('records')
        },
        "error_examples": error_types
    }


def analyze_performance_by_source(results_data):
    """Analyze performance based on news sources."""
    df = pd.DataFrame(results_data)

    # Add 'correct' column
    df['correct'] = df['true_label'] == df['predicted_label']

    # Group by source
    if 'source' in df.columns:
        source_accuracy = df.groupby('source')['correct'].mean().reset_index()
        source_count = df.groupby('source').size().reset_index(name='count')

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='source', y='correct', data=source_accuracy)
        plt.title('Accuracy by News Source')
        plt.xlabel('Source')
        plt.ylabel('Classification Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "accuracy_by_source.png"))
        plt.close()

        return {
            "source_accuracy": source_accuracy.to_dict('records'),
            "source_count": source_count.to_dict('records')
        }
    else:
        return None


def analyze_title_length(results_data):
    """Analyze performance based on title length."""
    df = pd.DataFrame(results_data)
    df['correct'] = df['true_label'] == df['predicted_label']
    df['title_length'] = df['title'].apply(lambda x: len(str(x).split()))

    # Bin the title lengths
    df['length_bin'] = pd.cut(df['title_length'],
                              bins=[0, 5, 10, 15, 20, 25, float('inf')],
                              labels=['1-5', '6-10', '11-15', '16-20', '21-25', '25+'])

    # Calculate accuracy by bin
    length_accuracy = df.groupby('length_bin')['correct'].mean().reset_index()
    length_count = df.groupby('length_bin').size().reset_index(name='count')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='length_bin', y='correct', data=length_accuracy)
    plt.title('Accuracy by Title Length')
    plt.xlabel('Title Length (words)')
    plt.ylabel('Classification Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "accuracy_by_length.png"))
    plt.close()

    return {
        "length_accuracy": length_accuracy.to_dict('records'),
        "length_count": length_count.to_dict('records')
    }


def compute_cross_dataset_performance(best_strategy_name):
    """Evaluate performance across different datasets (train, test, valid)."""
    print("Computing cross-dataset performance...")

    # Load datasets
    train_data = load_data(Config.TRAIN_FILE)
    test_data = load_data(Config.TEST_FILE)
    valid_data = load_data(Config.VALID_FILE)

    # Preprocess datasets
    processed_train = preprocess_data(train_data)
    processed_test = preprocess_data(test_data)
    processed_valid = preprocess_data(valid_data)

    # Sample from each dataset for evaluation
    sample_size = Config.SAMPLE_SIZE or 100  # Use a smaller sample if none specified

    # Evaluate on each dataset
    train_results, train_metrics, _ = run_evaluation(
        processed_train, best_strategy_name, sample_size
    )
    test_results, test_metrics, _ = run_evaluation(
        processed_test, best_strategy_name, sample_size
    )
    valid_results, valid_metrics, _ = run_evaluation(
        processed_valid, best_strategy_name, sample_size
    )

    # Compile results
    cross_dataset_results = {
        "train": {
            "accuracy": train_metrics["accuracy"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "f1": train_metrics["f1"]
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"]
        },
        "valid": {
            "accuracy": valid_metrics["accuracy"],
            "precision": valid_metrics["precision"],
            "recall": valid_metrics["recall"],
            "f1": valid_metrics["f1"]
        }
    }

    # Visualize cross-dataset performance
    datasets = ["train", "test", "valid"]
    accuracy = [cross_dataset_results[d]["accuracy"] for d in datasets]
    precision = [cross_dataset_results[d]["precision"] for d in datasets]
    recall = [cross_dataset_results[d]["recall"] for d in datasets]
    f1 = [cross_dataset_results[d]["f1"] for d in datasets]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.2

    plt.bar(x - width * 1.5, accuracy, width, label='Accuracy')
    plt.bar(x - width / 2, precision, width, label='Precision')
    plt.bar(x + width / 2, recall, width, label='Recall')
    plt.bar(x + width * 1.5, f1, width, label='F1')

    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Datasets')
    plt.xticks(x, datasets)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "cross_dataset_performance.png"))
    plt.close()

    return cross_dataset_results


def analyze_domain_performance(results_data):
    """Analyze performance based on news domains."""
    df = pd.DataFrame(results_data)
    df['correct'] = df['true_label'] == df['predicted_label']

    if 'domain' not in df.columns:
        return None

    # Count occurrences of each domain
    domain_counts = df['domain'].value_counts()

    # Get domains with at least 5 samples
    common_domains = domain_counts[domain_counts >= 5].index

    if len(common_domains) > 0:
        common_domains_df = df[df['domain'].isin(common_domains)]
        domain_accuracy = common_domains_df.groupby('domain')['correct'].mean().reset_index()
        domain_accuracy = domain_accuracy.sort_values('correct', ascending=False)

        # Plot top and bottom 10 domains by accuracy
        plt.figure(figsize=(12, 8))
        top_domains = domain_accuracy.head(10)
        bottom_domains = domain_accuracy.tail(10)

        plt.subplot(1, 2, 1)
        sns.barplot(x='correct', y='domain', data=top_domains)
        plt.title('Top 10 Domains by Accuracy')
        plt.xlabel('Accuracy')

        plt.subplot(1, 2, 2)
        sns.barplot(x='correct', y='domain', data=bottom_domains)
        plt.title('Bottom 10 Domains by Accuracy')
        plt.xlabel('Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "accuracy_by_domain.png"))
        plt.close()

        return {
            "top_domains": top_domains.to_dict('records'),
            "bottom_domains": bottom_domains.to_dict('records')
        }
    else:
        return None


def generate_final_report(metrics, efficiency, error_analysis, source_analysis, length_analysis, domain_analysis=None,
                          cross_dataset=None):
    """Generate a comprehensive evaluation report."""
    report = {
        "metrics": metrics,
        "efficiency": efficiency,
        "error_analysis": error_analysis,
        "source_analysis": source_analysis,
        "length_analysis": length_analysis
    }

    if domain_analysis:
        report["domain_analysis"] = domain_analysis

    if cross_dataset:
        report["cross_dataset_performance"] = cross_dataset

    # Save full report to JSON
    with open(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "evaluation_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Generate a text summary
    with open(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "evaluation_summary.txt"), 'w') as f:
        f.write("FAKENEWSNET DATASET - EVALUATION SUMMARY\n")
        f.write("======================================\n\n")

        f.write("CLASSIFICATION METRICS\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")

        f.write("EFFICIENCY METRICS\n")
        f.write(f"Average tokens per title: {efficiency['avg_tokens_per_title']:.2f}\n")
        f.write(f"Average processing time: {efficiency['avg_processing_time']:.2f} seconds\n")
        f.write(f"Total tokens used: {efficiency['total_tokens']}\n")
        f.write(f"Estimated cost: ${efficiency['estimated_cost']:.4f}\n\n")

        # Add cross-dataset performance if available
        if cross_dataset:
            f.write("CROSS-DATASET PERFORMANCE\n")
            for dataset in cross_dataset:
                f.write(f"{dataset.upper()} - F1: {cross_dataset[dataset]['f1']:.4f}, ")
                f.write(f"Accuracy: {cross_dataset[dataset]['accuracy']:.4f}\n")
            f.write("\n")

        f.write("SOURCE PERFORMANCE\n")
        if source_analysis and 'source_accuracy' in source_analysis:
            for record in source_analysis['source_accuracy']:
                source = record['source']
                accuracy = record['correct']
                f.write(f"{source}: {accuracy:.4f}\n")
        f.write("\n")

        f.write("TITLE LENGTH ANALYSIS\n")
        if length_analysis and 'length_accuracy' in length_analysis:
            for record in length_analysis['length_accuracy']:
                length_bin = record['length_bin']
                accuracy = record['correct']
                f.write(f"Titles with {length_bin} words: {accuracy:.4f}\n")
        f.write("\n")

        f.write("ERROR ANALYSIS\n")
        for label, count in zip(
                [e["true_label"] for e in error_analysis["error_counts"]["by_true_label"]],
                [e["count"] for e in error_analysis["error_counts"]["by_true_label"]]
        ):
            f.write(f"Errors for true label '{label}': {count}\n")

        f.write("\nFor more details, see the full JSON report and visualization files.")

    return report


def main():
    # Create directories if they don't exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.FAKENEWSNET_RESULTS_DIR, exist_ok=True)

    # Load the best strategy if available
    best_strategy = load_best_strategy()
    print(f"Using prompt strategy: {best_strategy}")

    # Load and preprocess test data
    print("Loading test data...")
    test_data = load_data(Config.TEST_FILE)
    processed_test = preprocess_data(test_data)

    # Run evaluation
    print("Running evaluation...")
    results, metrics, efficiency = run_evaluation(
        processed_test, best_strategy, sample_size=Config.SAMPLE_SIZE
    )

    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, ["fake", "real"], "Test Set Confusion Matrix",
        os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "confusion_matrix.png")
    )

    # Analyze errors
    print("Analyzing errors...")
    error_analysis = analyze_errors(results)

    # Analyze performance by source
    print("Analyzing performance by source...")
    source_analysis = analyze_performance_by_source(results)

    # Analyze performance by title length
    print("Analyzing performance by title length...")
    length_analysis = analyze_title_length(results)

    # Analyze performance by domain
    print("Analyzing performance by domain...")
    domain_analysis = analyze_domain_performance(results)

    # Optional: Compute cross-dataset performance
    cross_dataset = None
    if Config.SAMPLE_SIZE and Config.SAMPLE_SIZE <= 200:
        # Only run cross-dataset if we're using a small sample size
        print("Computing cross-dataset performance...")
        cross_dataset = compute_cross_dataset_performance(best_strategy)

    # Generate final report
    print("Generating final report...")
    report = generate_final_report(
        metrics, efficiency, error_analysis, source_analysis, length_analysis,
        domain_analysis, cross_dataset
    )

    # Print summary results
    print("\nEvaluation Results Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average tokens per title: {efficiency['avg_tokens_per_title']:.2f}")
    print(f"Estimated cost: ${efficiency['estimated_cost']:.4f}")

    print("\nEvaluation complete! Results saved to:", Config.FAKENEWSNET_RESULTS_DIR)


if __name__ == "__main__":
    main()