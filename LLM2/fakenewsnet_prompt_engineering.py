import os
import pandas as pd
import numpy as np
import time
import json
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()


# Configuration
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

    # Model settings
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4"
    TEMPERATURE = 0.0  # Lower temperature for more deterministic outputs

    # Test sample size for prompt testing
    TEST_SAMPLE_SIZE = 50


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

    return processed_data[relevant_columns]


# Define different prompt strategies
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
            max_tokens=10  # Keep this small for efficiency - we only need one word
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


def evaluate_prompt_strategy(data, strategy_name, sample_size=None):
    """Evaluate a prompt strategy on the dataset."""
    if sample_size and len(data) > sample_size:
        eval_data = data.sample(sample_size, random_state=42)
    else:
        eval_data = data

    results = []
    true_labels = []
    predicted_labels = []

    total_tokens = 0
    total_time = 0

    for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc=f"Evaluating {strategy_name}"):
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
        "f1": f1_score(true_labels, predicted_labels, pos_label="real", average="binary")
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


def compare_strategies(data, sample_size=Config.TEST_SAMPLE_SIZE):
    """Compare different prompt strategies."""
    comparison = {}

    for strategy_name in prompt_strategies.keys():
        print(f"Evaluating strategy: {strategy_name}")
        results, metrics, efficiency = evaluate_prompt_strategy(
            data, strategy_name, sample_size
        )

        comparison[strategy_name] = {
            "metrics": metrics,
            "efficiency": efficiency
        }

    # Create comparison visualizations
    strategies = list(comparison.keys())
    accuracies = [comparison[s]["metrics"]["accuracy"] for s in strategies]
    f1_scores = [comparison[s]["metrics"]["f1"] for s in strategies]
    token_usage = [comparison[s]["efficiency"]["avg_tokens_per_title"] for s in strategies]

    # Accuracy and F1 comparison
    plt.figure(figsize=(12, 6))

    x = np.arange(len(strategies))
    width = 0.35

    plt.bar(x - width / 2, accuracies, width, label='Accuracy')
    plt.bar(x + width / 2, f1_scores, width, label='F1 Score')

    plt.xlabel('Prompt Strategy')
    plt.ylabel('Score')
    plt.title('Accuracy and F1 Score by Prompt Strategy')
    plt.xticks(x, strategies)
    plt.legend()
    plt.tight_layout()

    os.makedirs(Config.FAKENEWSNET_RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "prompt_comparison.png"))
    plt.close()

    # Token usage comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=strategies, y=token_usage)
    plt.xlabel('Prompt Strategy')
    plt.ylabel('Average Tokens per Title')
    plt.title('Token Usage by Prompt Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "token_usage.png"))
    plt.close()

    # Save comparison data
    with open(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "prompt_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def analyze_by_source(data, best_strategy, sample_size=100):
    """Analyze performance of the best strategy by source."""
    # Evaluate on GossipCop data
    gossipcop_data = data[data['source'] == 'gossipcop'].sample(
        min(sample_size, len(data[data['source'] == 'gossipcop'])), random_state=42)
    gossipcop_results, gossipcop_metrics, _ = evaluate_prompt_strategy(gossipcop_data, best_strategy)

    # Evaluate on PolitiFact data
    politifact_data = data[data['source'] == 'politifact'].sample(
        min(sample_size, len(data[data['source'] == 'politifact'])), random_state=42)
    politifact_results, politifact_metrics, _ = evaluate_prompt_strategy(politifact_data, best_strategy)

    # Compile results
    source_results = {
        "gossipcop": gossipcop_metrics,
        "politifact": politifact_metrics
    }

    # Visualize
    sources = ["gossipcop", "politifact"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    plt.figure(figsize=(10, 6))

    values = {}
    for metric in metrics:
        values[metric] = [source_results[source][metric] for source in sources]

    x = np.arange(len(sources))
    width = 0.2

    plt.bar(x - width * 1.5, values["accuracy"], width, label='Accuracy')
    plt.bar(x - width / 2, values["precision"], width, label='Precision')
    plt.bar(x + width / 2, values["recall"], width, label='Recall')
    plt.bar(x + width * 1.5, values["f1"], width, label='F1')

    plt.xlabel('News Source')
    plt.ylabel('Score')
    plt.title(f'Performance Metrics by Source with {best_strategy} Strategy')
    plt.xticks(x, sources)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "source_performance.png"))
    plt.close()

    return source_results


def analyze_errors(results_data, best_strategy):
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

    # Find examples of each error type
    error_types = {}

    # False positives (fake classified as real)
    false_positives = errors[(errors['true_label'] == 'fake') & (errors['predicted_label'] == 'real')]
    error_types['false_positives'] = false_positives.head(5).to_dict('records')

    # False negatives (real classified as fake)
    false_negatives = errors[(errors['true_label'] == 'real') & (errors['predicted_label'] == 'fake')]
    error_types['false_negatives'] = false_negatives.head(5).to_dict('records')

    # Save detailed error analysis
    with open(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, f"error_analysis_{best_strategy}.json"), 'w') as f:
        json.dump({
            "error_counts": {
                "by_true_label": error_by_true_label.to_dict('records'),
                "by_predicted_label": error_by_pred_label.to_dict('records')
            },
            "error_examples": error_types
        }, f, indent=2)

    return error_types


def analyze_title_length_impact(results_data):
    """Analyze how title length affects performance."""
    df = pd.DataFrame(results_data)

    # Calculate title length
    df['title_length'] = df['title'].apply(lambda x: len(x.split()))
    df['correct'] = df['true_label'] == df['predicted_label']

    # Bin by title length
    df['length_bin'] = pd.cut(df['title_length'],
                              bins=[0, 5, 10, 15, 20, 25, float('inf')],
                              labels=['1-5', '6-10', '11-15', '16-20', '21-25', '25+'])

    # Calculate accuracy by length bin
    length_accuracy = df.groupby('length_bin')['correct'].mean().reset_index()
    length_count = df.groupby('length_bin').size().reset_index(name='count')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='length_bin', y='correct', data=length_accuracy)
    plt.xlabel('Title Length (words)')
    plt.ylabel('Classification Accuracy')
    plt.title('Accuracy by Title Length')
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "accuracy_by_length.png"))
    plt.close()

    return {
        "accuracy_by_length": length_accuracy.to_dict('records'),
        "count_by_length": length_count.to_dict('records')
    }


def main():
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.FAKENEWSNET_RESULTS_DIR, exist_ok=True)

    print("Loading validation data...")
    valid_data = load_data(Config.VALID_FILE)

    print("Preprocessing data...")
    processed_valid = preprocess_data(valid_data)

    print("Comparing prompt strategies...")
    strategy_comparison = compare_strategies(processed_valid)

    # Output best strategy
    best_strategy = max(strategy_comparison.keys(), key=lambda s: strategy_comparison[s]["metrics"]["f1"])
    best_accuracy = strategy_comparison[best_strategy]["metrics"]["accuracy"]
    best_f1 = strategy_comparison[best_strategy]["metrics"]["f1"]
    best_token_usage = strategy_comparison[best_strategy]["efficiency"]["avg_tokens_per_title"]
    best_cost = strategy_comparison[best_strategy]["efficiency"]["estimated_cost"]

    print("\nBest Prompt Strategy Results:")
    print(f"Strategy: {best_strategy}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Average Tokens: {best_token_usage:.1f}")
    print(f"Estimated Cost: ${best_cost:.4f}")

    # Analyze performance by source
    print("\nAnalyzing performance by source...")
    source_results = analyze_by_source(processed_valid, best_strategy)

    print("GossipCop Accuracy:", source_results["gossipcop"]["accuracy"])
    print("PolitiFact Accuracy:", source_results["politifact"]["accuracy"])

    # Evaluate best strategy on full validation set for error analysis
    print("\nEvaluating best strategy for error analysis...")
    best_results, _, _ = evaluate_prompt_strategy(processed_valid, best_strategy,
                                                  sample_size=Config.TEST_SAMPLE_SIZE * 2)

    # Perform error analysis
    print("Analyzing errors...")
    error_analysis = analyze_errors(best_results, best_strategy)

    # Analyze title length impact
    print("Analyzing title length impact...")
    length_analysis = analyze_title_length_impact(best_results)

    # Save the best strategy for use in the evaluation script
    with open(os.path.join(Config.FAKENEWSNET_RESULTS_DIR, "best_strategy.json"), 'w') as f:
        json.dump({
            "strategy_name": best_strategy,
            "metrics": strategy_comparison[best_strategy]["metrics"],
            "efficiency": strategy_comparison[best_strategy]["efficiency"],
            "source_performance": source_results,
            "title_length_analysis": length_analysis
        }, f, indent=2)

    print("\nOptimization complete. Results saved to:", Config.FAKENEWSNET_RESULTS_DIR)


if __name__ == "__main__":
    main()