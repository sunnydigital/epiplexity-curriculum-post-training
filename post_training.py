from datasets import load_dataset

def main():
    # Load the dataset
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Print the first few examples
    print(dataset["train"][:5])

if __name__ == "__main__":
    main()