import os
import argparse
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='FakeNewsNet-Based Fake News Detection')
    parser.add_argument('--task', type=str, default='evaluate',
                        choices=['preprocess', 'optimize_prompts', 'evaluate', 'all'],
                        help='Task to perform')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        help='OpenAI model to use')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of examples to use (for optimization and evaluation)')

    args = parser.parse_args()

    # Set API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Import appropriate modules based on task
    if args.task == 'preprocess' or args.task == 'all':
        print("Running data preprocessing for FakeNewsNet...")
        import fakenewsnet_preprocessing
        fakenewsnet_preprocessing.main()

    if args.task == 'optimize_prompts' or args.task == 'all':
        print("\nOptimizing prompts for FakeNewsNet...")
        import fakenewsnet_prompt_engineering
        # Update sample size and model
        fakenewsnet_prompt_engineering.Config.TEST_SAMPLE_SIZE = args.sample_size
        fakenewsnet_prompt_engineering.Config.MODEL_NAME = args.model
        fakenewsnet_prompt_engineering.main()

    if args.task == 'evaluate' or args.task == 'all':
        print("\nRunning evaluation for FakeNewsNet...")
        import fakenewsnet_evaluation
        # Update sample size and model
        fakenewsnet_evaluation.Config.SAMPLE_SIZE = args.sample_size
        fakenewsnet_evaluation.Config.MODEL_NAME = args.model
        fakenewsnet_evaluation.main()

    print("All FakeNewsNet tasks completed!")


if __name__ == "__main__":
    main()