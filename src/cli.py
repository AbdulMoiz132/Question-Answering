"""
Command Line Interface for Question Answering System
Allows interactive questioning with context input
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basic_qa_offline import simple_qa_system
from basic_qa import answer_question, check_internet_connection, setup_transformer_qa_model
from data_loader import load_data
from evaluator import evaluate_qa_system, print_evaluation_results
import pandas as pd

def interactive_qa_session(qa_function):
    """
    Run an interactive QA session
    """
    print("\n" + "="*60)
    print("INTERACTIVE QUESTION ANSWERING SESSION")
    print("="*60)
    print("Enter 'quit' to exit")
    print("Enter 'new' to input a new context")
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
        print("\nEnter your question (or 'new' for new context, 'quit' to exit):")
        question = input("Q: ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'new':
            context = ""
            continue
        elif not question:
            print("Question cannot be empty!")
            continue
        
        # Get answer
        try:
            result = qa_function(None, context, question)
            if isinstance(result, dict):
                answer = result.get('answer', 'No answer found')
                confidence = result.get('score', 0.0)
                method = result.get('method', 'unknown')
                print(f"A: {answer}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Method: {method}")
            else:
                print(f"A: {result}")
        except Exception as e:
            print(f"Error: {e}")

def qa_with_sample_data(qa_function):
    """
    Run QA with the sample dataset
    """
    print("\n" + "="*60)
    print("QUESTION ANSWERING WITH SAMPLE DATA")
    print("="*60)
    
    # Load sample data
    data, _ = load_data(use_online=False)
    
    print(f"Loaded {len(data)} sample questions")
    print("\nAnswering questions:")
    print("-" * 30)
    
    for _, row in data.iterrows():
        print(f"\nContext: {row['title']}")
        print(f"Q: {row['question']}")
        
        try:
            result = qa_function(None, row['context'], row['question'])
            if isinstance(result, dict):
                answer = result.get('answer', 'No answer found')
                confidence = result.get('score', 0.0)
                method = result.get('method', 'unknown')
                print(f"A: {answer}")
                print(f"Expected: {row['answer_text']}")
                print(f"Confidence: {confidence:.2f} | Method: {method}")
            else:
                print(f"A: {result}")
                print(f"Expected: {row['answer_text']}")
        except Exception as e:
            print(f"Error: {e}")

def evaluate_system(qa_function):
    """
    Evaluate the QA system and print results
    """
    print("\n" + "="*60)
    print("EVALUATING QA SYSTEM")
    print("="*60)
    
    # Load test data
    test_data, _ = load_data(use_online=False)
    
    # Create wrapper function for evaluation
    def eval_qa_function(context, question):
        result = qa_function(None, context, question)
        if isinstance(result, dict):
            return result.get('answer', 'No answer')
        return str(result)
    
    # Run evaluation
    results = evaluate_qa_system(eval_qa_function, test_data)
    print_evaluation_results(results)

def main():
    """
    Main CLI function
    """
    parser = argparse.ArgumentParser(description="Question Answering System CLI")
    parser.add_argument("--mode", choices=['interactive', 'sample', 'evaluate'], 
                       default='interactive',
                       help="Mode to run the CLI in")
    parser.add_argument("--use-transformers", action='store_true',
                       help="Try to use transformer models (requires internet)")
    
    args = parser.parse_args()
    
    print("Question Answering System CLI")
    print("="*50)
    
    # Setup QA function
    if args.use_transformers and check_internet_connection():
        print("Loading transformer model...")
        tokenizer, model, qa_pipeline = setup_transformer_qa_model()
        if qa_pipeline:
            qa_function = lambda pipeline, context, question: answer_question(qa_pipeline, context, question)
            print("✓ Using transformer model")
        else:
            qa_function = lambda pipeline, context, question: simple_qa_system(context, question)
            print("✓ Using rule-based system (transformer failed)")
    else:
        qa_function = lambda pipeline, context, question: simple_qa_system(context, question)
        print("✓ Using rule-based system")
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_qa_session(qa_function)
    elif args.mode == 'sample':
        qa_with_sample_data(qa_function)
    elif args.mode == 'evaluate':
        evaluate_system(qa_function)
    
    print("\nThank you for using the QA system!")

if __name__ == "__main__":
    main()
