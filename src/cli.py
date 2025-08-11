"""
Command Line Interface for Question Answering System
Enhanced with model management capabilities
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import create_enhanced_qa_system
from data_loader import load_data
from evaluator import evaluate_qa_system, print_evaluation_results
import pandas as pd

def interactive_qa_session(qa_manager):
    """
    Run an interactive QA session
    """
    print("\n" + "="*60)
    print("INTERACTIVE QUESTION ANSWERING SESSION")
    print("="*60)
    print("Enter 'quit' to exit")
    print("Enter 'new' to input a new context")
    print("Enter 'info' to see model information")
    print("="*60)
    
    context = ""
    
    while True:
        if not context:
            print("\nPlease enter a context (text passage):")
            context = input("> ").strip()
            if context.lower() == 'quit':
                break
            if not context:
                print("Context cannot be empty!")
                continue
        
        print(f"\nContext: {context[:100]}{'...' if len(context) > 100 else ''}")
        print("\nEnter your question (or 'new' for new context, 'info' for model info, 'quit' to exit):")
        question = input("Q: ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'new':
            context = ""
            continue
        elif question.lower() == 'info':
            info = qa_manager.get_model_info()
            print("Current model:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            continue
        elif not question:
            print("Question cannot be empty!")
            continue
        
        # Get answer
        try:
            result = qa_manager.predict(context, question)
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['score']:.2f}")
            print(f"Method: {result['method']}")
        except Exception as e:
            print(f"Error: {e}")

def qa_with_sample_data(qa_manager):
    """
    Run QA with the sample dataset
    """
    print("\n" + "="*60)
    print("QUESTION ANSWERING WITH SAMPLE DATA")
    print("="*60)
    
    # Load sample data
    data, _ = load_data(use_online=False)
    
    print(f"Loaded {len(data)} sample questions")
    print(f"Using model: {qa_manager.get_model_info()}")
    print("\nAnswering questions:")
    print("-" * 30)
    
    for _, row in data.iterrows():
        print(f"\nContext: {row['title']}")
        print(f"Q: {row['question']}")
        
        try:
            result = qa_manager.predict(row['context'], row['question'])
            print(f"A: {result['answer']}")
            print(f"Expected: {row['answer_text']}")
            print(f"Confidence: {result['score']:.2f} | Method: {result['method']}")
        except Exception as e:
            print(f"Error: {e}")

def evaluate_system(qa_manager):
    """
    Evaluate the QA system and print results
    """
    print("\n" + "="*60)
    print("EVALUATING QA SYSTEM")
    print("="*60)
    
    print(f"Using model: {qa_manager.get_model_info()}")
    
    # Load test data
    test_data, _ = load_data(use_online=False)
    
    # Create wrapper function for evaluation
    def eval_qa_function(context, question):
        result = qa_manager.predict(context, question)
        return result['answer']
    
    # Run evaluation
    results = evaluate_qa_system(eval_qa_function, test_data)
    print_evaluation_results(results)

def main():
    """
    Main CLI function
    """
    parser = argparse.ArgumentParser(description="Enhanced Question Answering System CLI")
    parser.add_argument("--mode", choices=['interactive', 'sample', 'evaluate', 'models'], 
                       default='interactive',
                       help="Mode to run the CLI in")
    
    args = parser.parse_args()
    
    print("Enhanced Question Answering System CLI")
    print("="*50)
    
    # Setup enhanced QA system with model management
    qa_manager = create_enhanced_qa_system()
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_qa_session(qa_manager)
    elif args.mode == 'sample':
        qa_with_sample_data(qa_manager)
    elif args.mode == 'evaluate':
        evaluate_system(qa_manager)
    elif args.mode == 'models':
        print("\nAvailable models:")
        for model in qa_manager.list_available_models():
            print(f"  - {model}")
        print(f"\nCurrent model: {qa_manager.get_model_info()}")
    
    print("\nThank you for using the Enhanced QA system!")

if __name__ == "__main__":
    main()
