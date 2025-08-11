"""
Data loader and preprocessor for Question Answering
Supports both SQuAD dataset (online) and sample data (offline)
"""

import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re

def create_sample_squad_data():
    """
    Create sample SQuAD-like data for offline development and testing
    """
    sample_data = {
        "data": [
            {
                "title": "Amazon_rainforest",
                "paragraphs": [
                    {
                        "context": "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.",
                        "qas": [
                            {
                                "id": "amazon_001",
                                "question": "What percentage of the Amazon rainforest is in Brazil?",
                                "answers": [
                                    {
                                        "text": "60%",
                                        "answer_start": 462
                                    }
                                ]
                            },
                            {
                                "id": "amazon_002", 
                                "question": "How many square kilometers does the Amazon basin cover?",
                                "answers": [
                                    {
                                        "text": "7,000,000 km2",
                                        "answer_start": 140
                                    }
                                ]
                            },
                            {
                                "id": "amazon_003",
                                "question": "Which country has the second largest portion of the rainforest?",
                                "answers": [
                                    {
                                        "text": "Peru",
                                        "answer_start": 507
                                    }
                                ]
                            },
                            {
                                "id": "amazon_004",
                                "question": "How many nations have territory in the Amazon region?",
                                "answers": [
                                    {
                                        "text": "nine nations",
                                        "answer_start": 347
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "title": "Artificial_intelligence",
                "paragraphs": [
                    {
                        "context": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.",
                        "qas": [
                            {
                                "id": "ai_001",
                                "question": "What does AI stand for?",
                                "answers": [
                                    {
                                        "text": "Artificial intelligence",
                                        "answer_start": 0
                                    }
                                ]
                            },
                            {
                                "id": "ai_002",
                                "question": "What are intelligent agents?",
                                "answers": [
                                    {
                                        "text": "any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals",
                                        "answer_start": 226
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return sample_data

def load_squad_dataset_online():
    """
    Load SQuAD dataset from Hugging Face (requires internet)
    """
    try:
        from datasets import load_dataset
        print("Loading SQuAD v1.1 dataset from Hugging Face...")
        
        # Load train and validation sets
        dataset = load_dataset("squad")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        
        print(f"✓ Loaded {len(train_data)} training examples")
        print(f"✓ Loaded {len(val_data)} validation examples")
        
        return train_data, val_data
    
    except Exception as e:
        print(f"Failed to load online SQuAD dataset: {e}")
        return None, None

def save_sample_data_to_files():
    """
    Save sample data to JSON files in data/ directory
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    sample_data = create_sample_squad_data()
    
    # Save as JSON file
    with open("data/sample_squad.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Sample SQuAD data saved to data/sample_squad.json")
    
    # Also create a CSV version for easier analysis
    examples = []
    for article in sample_data["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answer_text": qa["answers"][0]["text"],
                    "answer_start": qa["answers"][0]["answer_start"]
                })
    
    df = pd.DataFrame(examples)
    df.to_csv("data/sample_squad.csv", index=False)
    print("✓ Sample data also saved as CSV to data/sample_squad.csv")
    
    return df

def load_data(use_online=False):
    """
    Load QA data - tries online first, falls back to sample data
    
    Args:
        use_online (bool): Whether to attempt loading online SQuAD data
    
    Returns:
        tuple: (train_data, val_data) or (sample_df, None) for offline
    """
    if use_online:
        train_data, val_data = load_squad_dataset_online()
        if train_data is not None:
            return train_data, val_data
        else:
            print("Falling back to sample data...")
    
    print("Using offline sample data...")
    
    # Check if sample data exists
    if not os.path.exists("data/sample_squad.json"):
        print("Creating sample data...")
        sample_df = save_sample_data_to_files()
    else:
        print("Loading existing sample data...")
        sample_df = pd.read_csv("data/sample_squad.csv")
    
    return sample_df, None

def preprocess_data(data, is_dataframe=True):
    """
    Preprocess the data for training/evaluation
    
    Args:
        data: Either pandas DataFrame (sample) or HF dataset (online)
        is_dataframe (bool): Whether data is a pandas DataFrame
    
    Returns:
        list: Processed examples
    """
    examples = []
    
    if is_dataframe:
        # Process DataFrame (sample data)
        for _, row in data.iterrows():
            examples.append({
                "id": row["id"],
                "title": row["title"],
                "context": row["context"],
                "question": row["question"],
                "answer_text": row["answer_text"],
                "answer_start": row["answer_start"]
            })
    else:
        # Process HuggingFace dataset (online data)
        for example in data:
            examples.append({
                "id": example["id"],
                "title": example["title"],
                "context": example["context"],
                "question": example["question"],
                "answer_text": example["answers"]["text"][0],
                "answer_start": example["answers"]["answer_start"][0]
            })
    
    print(f"✓ Preprocessed {len(examples)} examples")
    return examples

def analyze_data(examples):
    """
    Analyze the dataset and print statistics
    """
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    print(f"Total examples: {len(examples)}")
    
    # Question length analysis
    question_lengths = [len(ex["question"].split()) for ex in examples]
    print(f"Average question length: {sum(question_lengths)/len(question_lengths):.1f} words")
    print(f"Question length range: {min(question_lengths)} - {max(question_lengths)} words")
    
    # Context length analysis
    context_lengths = [len(ex["context"].split()) for ex in examples]
    print(f"Average context length: {sum(context_lengths)/len(context_lengths):.1f} words")
    print(f"Context length range: {min(context_lengths)} - {max(context_lengths)} words")
    
    # Answer length analysis
    answer_lengths = [len(ex["answer_text"].split()) for ex in examples]
    print(f"Average answer length: {sum(answer_lengths)/len(answer_lengths):.1f} words")
    print(f"Answer length range: {min(answer_lengths)} - {max(answer_lengths)} words")
    
    # Show sample questions
    print(f"\nSample questions:")
    for i, ex in enumerate(examples[:3]):
        print(f"{i+1}. {ex['question']}")
        print(f"   Answer: {ex['answer_text']}")
    
    print("="*50)

def main():
    """
    Main function to demonstrate data loading and preprocessing
    """
    print("Question Answering Data Preparation")
    print("="*50)
    
    # Try to load data (will use sample data due to connectivity)
    train_data, val_data = load_data(use_online=False)
    
    # Preprocess the data
    if isinstance(train_data, pd.DataFrame):
        # Sample data
        examples = preprocess_data(train_data, is_dataframe=True)
        analyze_data(examples)
    else:
        # Online data
        train_examples = preprocess_data(train_data, is_dataframe=False)
        val_examples = preprocess_data(val_data, is_dataframe=False)
        
        print("Training data:")
        analyze_data(train_examples[:100])  # Analyze first 100 examples
        
        print("\nValidation data:")
        analyze_data(val_examples[:50])   # Analyze first 50 examples

if __name__ == "__main__":
    main()
