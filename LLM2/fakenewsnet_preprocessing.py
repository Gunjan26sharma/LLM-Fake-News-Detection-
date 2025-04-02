import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import json

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Configuration class
class Config:
    # Paths
    DATA_DIR = "data/"
    RESULTS_DIR = "results/"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")

    # Dataset files for FakeNewsNet
    GOSSIPCOP_FAKE_FILE = os.path.join(DATA_DIR, "gossipcop_fake.csv")
    GOSSIPCOP_REAL_FILE = os.path.join(DATA_DIR, "gossipcop_real.csv")
    POLITIFACT_FAKE_FILE = os.path.join(DATA_DIR, "politifact_fake.csv")
    POLITIFACT_REAL_FILE = os.path.join(DATA_DIR, "politifact_real.csv")

    # Output processed files
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed/")
    TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
    TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")
    VALID_FILE = os.path.join(PROCESSED_DIR, "valid.csv")

    # Binary labels
    LABEL_MAPPING = {
        "fake": 0,
        "real": 1
    }

    # Source mapping
    SOURCE_MAPPING = {
        "gossipcop": 0,
        "politifact": 1
    }


def load_data():
    """Load and combine the FakeNewsNet datasets."""
    print("Loading FakeNewsNet datasets...")

    # Create directories if they don't exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    # Load GossipCop data
    gc_fake = pd.read_csv(Config.GOSSIPCOP_FAKE_FILE)
    gc_real = pd.read_csv(Config.GOSSIPCOP_REAL_FILE)

    # Load PolitiFact data
    pf_fake = pd.read_csv(Config.POLITIFACT_FAKE_FILE)
    pf_real = pd.read_csv(Config.POLITIFACT_REAL_FILE)

    # Add source and label columns
    gc_fake['source'] = 'gossipcop'
    gc_fake['label'] = 'fake'

    gc_real['source'] = 'gossipcop'
    gc_real['label'] = 'real'

    pf_fake['source'] = 'politifact'
    pf_fake['label'] = 'fake'

    pf_real['source'] = 'politifact'
    pf_real['label'] = 'real'

    # Combine all datasets
    combined_data = pd.concat([gc_fake, gc_real, pf_fake, pf_real], ignore_index=True)

    print(f"Total samples: {len(combined_data)}")
    print(f"GossipCop fake: {len(gc_fake)}, GossipCop real: {len(gc_real)}")
    print(f"PolitiFact fake: {len(pf_fake)}, PolitiFact real: {len(pf_real)}")

    return combined_data


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(data, clean_titles=True):
    """Clean and preprocess the data."""
    # Create a copy to avoid modifying the original
    processed_data = data.copy()

    # Basic cleaning
    if clean_titles:
        processed_data['title'] = processed_data['title'].apply(clean_text)

    # Handle missing values
    processed_data['title'] = processed_data['title'].fillna("")
    processed_data['news_url'] = processed_data['news_url'].fillna("")

    # Extract domain from URL
    processed_data['domain'] = processed_data['news_url'].apply(
        lambda url: url.split('/')[2] if '//' in url and len(url.split('/')) > 2 else "")

    # Count the number of tweets for each news
    processed_data['tweet_count'] = processed_data['tweet_ids'].apply(
        lambda x: len(str(x).split('\t')) if pd.notna(x) else 0)

    # Map labels and source to numeric
    processed_data['label_num'] = processed_data['label'].map(Config.LABEL_MAPPING)
    processed_data['source_num'] = processed_data['source'].map(Config.SOURCE_MAPPING)

    # Calculate title length
    processed_data['title_length'] = processed_data['title'].apply(len)
    processed_data['title_word_count'] = processed_data['title'].apply(lambda x: len(x.split()))

    return processed_data


def split_data(data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, stratify_by='label'):
    """Split the data into train, validation, and test sets, stratified by label."""
    from sklearn.model_selection import train_test_split

    # First split: separate train and temp (validation + test) data
    train_data, temp_data = train_test_split(
        data,
        test_size=(valid_ratio + test_ratio),
        random_state=42,
        stratify=data[stratify_by]
    )

    # Second split: separate validation and test from temp
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - valid_ratio_adjusted),
        random_state=42,
        stratify=temp_data[stratify_by]
    )

    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(valid_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Save to CSV files
    train_data.to_csv(Config.TRAIN_FILE, index=False)
    valid_data.to_csv(Config.VALID_FILE, index=False)
    test_data.to_csv(Config.TEST_FILE, index=False)

    return train_data, valid_data, test_data


def analyze_label_distribution(data, title="Label Distribution by Source"):
    """Analyze and plot the distribution of labels by source."""
    plt.figure(figsize=(10, 6))

    # Count by source and label
    label_source_counts = data.groupby(['source', 'label']).size().unstack()

    # Plot stacked bar chart
    label_source_counts.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel("Source")
    plt.ylabel("Count")
    plt.legend(title="Label")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "label_source_distribution.png"))
    plt.close()

    return label_source_counts


def analyze_title_length(data):
    """Analyze and plot the distribution of title lengths."""
    plt.figure(figsize=(12, 8))

    # Plot by label
    sns.histplot(data=data, x='title_word_count', hue='label', bins=30, element="step")
    plt.title("Distribution of Title Lengths by Label")
    plt.xlabel("Number of Words in Title")
    plt.ylabel("Count")
    plt.axvline(data['title_word_count'].mean(), color='red', linestyle='--',
                label=f"Mean: {data['title_word_count'].mean():.2f}")
    plt.axvline(data['title_word_count'].median(), color='green', linestyle='--',
                label=f"Median: {data['title_word_count'].median():.2f}")
    plt.legend()
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "title_length_dist.png"))
    plt.close()

    # Return statistics
    length_stats = {
        "mean": data['title_word_count'].mean(),
        "median": data['title_word_count'].median(),
        "min": data['title_word_count'].min(),
        "max": data['title_word_count'].max(),
        "95th_percentile": data['title_word_count'].quantile(0.95)
    }

    return length_stats


def analyze_tweet_counts(data):
    """Analyze and plot the distribution of tweet counts."""
    plt.figure(figsize=(12, 8))

    # Get the 99th percentile for better visualization (outliers can skew the plot)
    upper_limit = data['tweet_count'].quantile(0.99)
    filtered_data = data[data['tweet_count'] <= upper_limit]

    # Plot by label
    sns.boxplot(data=filtered_data, x='label', y='tweet_count')
    plt.title("Distribution of Tweet Counts by Label")
    plt.xlabel("Label")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "tweet_count_dist.png"))
    plt.close()

    # Statistics
    tweet_stats = {
        "mean_fake": data[data['label'] == 'fake']['tweet_count'].mean(),
        "mean_real": data[data['label'] == 'real']['tweet_count'].mean(),
        "median_fake": data[data['label'] == 'fake']['tweet_count'].median(),
        "median_real": data[data['label'] == 'real']['tweet_count'].median(),
        "max_fake": data[data['label'] == 'fake']['tweet_count'].max(),
        "max_real": data[data['label'] == 'real']['tweet_count'].max()
    }

    return tweet_stats


def analyze_domains(data):
    """Analyze the distribution of domains by label."""
    # Count domains by label
    domain_fake = Counter(data[data['label'] == 'fake']['domain'])
    domain_real = Counter(data[data['label'] == 'real']['domain'])

    # Get top domains
    top_fake_domains = domain_fake.most_common(15)
    top_real_domains = domain_real.most_common(15)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(1, 2, 1)
    domains, counts = zip(*top_fake_domains)
    sns.barplot(x=list(counts), y=list(domains))
    plt.title("Top 15 Domains for Fake News")
    plt.xlabel("Count")
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    domains, counts = zip(*top_real_domains)
    sns.barplot(x=list(counts), y=list(domains))
    plt.title("Top 15 Domains for Real News")
    plt.xlabel("Count")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "domain_distribution.png"))
    plt.close()

    return {"top_fake_domains": top_fake_domains, "top_real_domains": top_real_domains}


def analyze_common_words(data, n=20):
    """Analyze and plot the most common words in titles by label."""
    stop_words = set(stopwords.words('english'))

    # For fake news
    fake_titles = ' '.join(data[data['label'] == 'fake']['title'])
    fake_words = word_tokenize(fake_titles)
    fake_filtered_words = [word.lower() for word in fake_words if word.lower() not in stop_words and word.isalpha()]
    fake_word_counts = Counter(fake_filtered_words)

    # For real news
    real_titles = ' '.join(data[data['label'] == 'real']['title'])
    real_words = word_tokenize(real_titles)
    real_filtered_words = [word.lower() for word in real_words if word.lower() not in stop_words and word.isalpha()]
    real_word_counts = Counter(real_filtered_words)

    # Get most common words
    fake_most_common = fake_word_counts.most_common(n)
    real_most_common = real_word_counts.most_common(n)

    # Plot
    plt.figure(figsize=(14, 10))

    plt.subplot(1, 2, 1)
    words, counts = zip(*fake_most_common)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {n} Words in Fake News Titles")
    plt.xlabel("Count")
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    words, counts = zip(*real_most_common)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {n} Words in Real News Titles")
    plt.xlabel("Count")
    plt.tight_layout()

    # Save figure
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.FIGURES_DIR, "common_words.png"))
    plt.close()

    return {"fake_most_common": fake_most_common, "real_most_common": real_most_common}


def generate_summary_report(data, length_stats, tweet_stats, domain_analysis, word_analysis):
    """Generate a summary report of the dataset."""
    summary = {
        "dataset_size": {
            "total": len(data),
            "fake": len(data[data['label'] == 'fake']),
            "real": len(data[data['label'] == 'real']),
            "gossipcop": len(data[data['source'] == 'gossipcop']),
            "politifact": len(data[data['source'] == 'politifact'])
        },
        "title_length_stats": length_stats,
        "tweet_count_stats": tweet_stats,
        "top_domains": {
            "fake": domain_analysis["top_fake_domains"][:5],
            "real": domain_analysis["top_real_domains"][:5]
        },
        "common_words": {
            "fake": word_analysis["fake_most_common"][:10],
            "real": word_analysis["real_most_common"][:10]
        }
    }

    # Save summary as JSON
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(Config.RESULTS_DIR, "dataset_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save text summary
    with open(os.path.join(Config.RESULTS_DIR, "dataset_summary.txt"), 'w') as f:
        f.write("FakeNewsNet Dataset Summary\n")
        f.write("==========================\n\n")
        f.write(f"Total samples: {summary['dataset_size']['total']}\n")
        f.write(f"Fake news: {summary['dataset_size']['fake']} samples\n")
        f.write(f"Real news: {summary['dataset_size']['real']} samples\n\n")

        f.write(f"GossipCop samples: {summary['dataset_size']['gossipcop']}\n")
        f.write(f"PolitiFact samples: {summary['dataset_size']['politifact']}\n\n")

        f.write("Title Length Statistics:\n")
        f.write(f"Average words per title: {length_stats['mean']:.2f}\n")
        f.write(f"Median words per title: {length_stats['median']:.0f}\n")
        f.write(f"Min words per title: {length_stats['min']:.0f}\n")
        f.write(f"Max words per title: {length_stats['max']:.0f}\n\n")

        f.write("Tweet Count Statistics:\n")
        f.write(f"Average tweets for fake news: {tweet_stats['mean_fake']:.2f}\n")
        f.write(f"Average tweets for real news: {tweet_stats['mean_real']:.2f}\n")
        f.write(f"Median tweets for fake news: {tweet_stats['median_fake']:.0f}\n")
        f.write(f"Median tweets for real news: {tweet_stats['median_real']:.0f}\n\n")

        f.write("Top Domains for Fake News:\n")
        for domain, count in summary['top_domains']['fake']:
            f.write(f"- {domain}: {count} samples\n")
        f.write("\n")

        f.write("Top Domains for Real News:\n")
        for domain, count in summary['top_domains']['real']:
            f.write(f"- {domain}: {count} samples\n")
        f.write("\n")

        f.write("Most Common Words in Fake News Titles:\n")
        for word, count in summary['common_words']['fake']:
            f.write(f"- {word}: {count} occurrences\n")
        f.write("\n")

        f.write("Most Common Words in Real News Titles:\n")
        for word, count in summary['common_words']['real']:
            f.write(f"- {word}: {count} occurrences\n")

    return summary


def main():
    # Create necessary directories
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)

    print("Loading and processing FakeNewsNet data...")
    data = load_data()

    print("Preprocessing data...")
    processed_data = preprocess_data(data)

    print("Splitting data into train/validation/test sets...")
    train_data, valid_data, test_data = split_data(processed_data)

    print("Analyzing label distribution...")
    label_counts = analyze_label_distribution(processed_data)

    print("Analyzing title lengths...")
    length_stats = analyze_title_length(processed_data)

    print("Analyzing tweet counts...")
    tweet_stats = analyze_tweet_counts(processed_data)

    print("Analyzing domains...")
    domain_analysis = analyze_domains(processed_data)

    print("Analyzing common words...")
    word_analysis = analyze_common_words(processed_data)

    print("Generating summary report...")
    summary = generate_summary_report(processed_data, length_stats, tweet_stats, domain_analysis, word_analysis)

    print("Analysis complete!")

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(processed_data)}")
    print(f"Fake samples: {len(processed_data[processed_data['label'] == 'fake'])}")
    print(f"Real samples: {len(processed_data[processed_data['label'] == 'real'])}")
    print(f"Average title length: {length_stats['mean']:.2f} words")
    print(f"Average tweets for fake news: {tweet_stats['mean_fake']:.2f}")
    print(f"Average tweets for real news: {tweet_stats['mean_real']:.2f}")


if __name__ == "__main__":
    main()