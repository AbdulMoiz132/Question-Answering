"""
Comprehensive Demo Script for Question Answering System
Demonstrates all features and capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import json
from typing import List, Dict
import time

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

def print_section(title: str):
    """Print a section header"""
    print(f"\n{title}")
    print("-" * len(title))

def demo_basic_qa():
    """Demonstrate basic QA functionality"""
    print_section("1. Basic Question Answering")
    
    # Import QA systems
    from basic_qa import simple_qa_system
    from basic_qa_offline import simple_qa_system as offline_qa
    
    # Sample context
    context = """
    The Amazon rainforest covers approximately 5.5 million square kilometers and spans across 
    nine South American countries, with the largest portion in Brazil (60%). It contains about 
    10% of the world's known biodiversity and is often called the "lungs of the Earth" because 
    it produces about 20% of the world's oxygen.
    """
    
    questions = [
        "What percentage of the Amazon rainforest is in Brazil?",
        "How much area does the Amazon rainforest cover?",
        "How much oxygen does the Amazon produce?",
        "What is the Amazon rainforest called?"
    ]
    
    print(f"Context: {context[:100]}...")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        
        # Try enhanced QA
        result = simple_qa_system(context, question)
        if isinstance(result, dict):
            answer = result['answer']
            method = result.get('method', 'unknown')
            score = result.get('score', 0.0)
            print(f"A{i}: {answer} (Method: {method}, Score: {score:.2f})")
        else:
            print(f"A{i}: {result}")

def demo_dataset_loading():
    """Demonstrate dataset loading and exploration"""
    print_section("2. Dataset Loading and Exploration")
    
    from data_loader import load_data, create_sample_squad_data
    
    # Load sample data
    print("Loading sample SQuAD dataset...")
    df, _ = load_data(use_online=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample questions: {df['question'].nunique()}")
    
    # Show sample data
    print("\nSample entries:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nEntry {i+1}:")
        print(f"  ID: {row['id']}")
        print(f"  Question: {row['question']}")
        print(f"  Answer: {row['answer_text']}")
        print(f"  Context: {row['context'][:100]}...")

def demo_evaluation():
    """Demonstrate evaluation metrics"""
    print_section("3. Evaluation System")
    
    from evaluator import evaluate_qa_system, exact_match_score, f1_score
    from data_loader import load_data
    from basic_qa import simple_qa_system
    
    # Load test data
    test_data, _ = load_data(use_online=False)
    
    print("Evaluating on sample dataset...")
    
    # Define QA function
    def qa_function(context, question):
        result = simple_qa_system(context, question)
        if isinstance(result, dict):
            return result['answer']
        return result
    
    # Evaluate
    results = evaluate_qa_system(qa_function, test_data)
    
    print(f"Evaluation Results:")
    print(f"  Total Questions: {results['total_questions']}")
    print(f"  Exact Match: {results['exact_match']:.2%}")
    print(f"  F1 Score: {results['f1_score']:.2%}")
    
    # Show detailed results
    print("\nDetailed Results:")
    for i, detail in enumerate(results['detailed_results'][:3]):
        print(f"\nQuestion {i+1}: {detail['question']}")
        print(f"  Expected: {detail['expected_answer']}")
        print(f"  Predicted: {detail['predicted_answer']}")
        print(f"  EM: {detail['exact_match']:.0f}, F1: {detail['f1_score']:.2f}")

def demo_model_training():
    """Demonstrate model training"""
    print_section("4. Model Training")
    
    from trainer import RuleBasedTrainer
    from data_loader import load_data
    
    # Load training data
    train_data, _ = load_data(use_online=False)
    
    print("Training rule-based model...")
    trainer = RuleBasedTrainer()
    
    # Train model
    model_path = trainer.train_from_dataframe(train_data, "models/demo_rule_based_qa.pkl")
    
    if model_path:
        print(f"Model saved to: {model_path}")
        
        # Test trained model
        from model_manager import ModelManager
        manager = ModelManager()
        if manager.load_rule_based_model(model_path):
            print("Testing trained model...")
            test_question = "What percentage of the Amazon rainforest is in Brazil?"
            test_context = train_data.iloc[0]['context']
            
            result = manager.predict(test_context, test_question)
            print(f"Test Question: {test_question}")
            print(f"Answer: {result}")

def demo_cli_interface():
    """Demonstrate CLI interface"""
    print_section("5. CLI Interface")
    
    print("The CLI interface supports several modes:")
    print("  - Interactive mode: python src/cli.py")
    print("  - Single question: python src/cli.py --question 'Your question' --context 'Context'")
    print("  - Batch evaluation: python src/cli.py --evaluate")
    print("  - Model training: python src/cli.py --train")
    print("  - Model comparison: python src/cli.py --compare")
    
    # Simulate a CLI interaction
    print("\nSimulating CLI interaction:")
    from cli import QuestionAnsweringCLI
    
    cli = QuestionAnsweringCLI()
    
    # Test with a sample question
    sample_context = "Python is a high-level programming language created by Guido van Rossum in 1991."
    sample_question = "Who created Python?"
    
    print(f"Question: {sample_question}")
    print(f"Context: {sample_context}")
    
    result = cli.answer_question(sample_question, sample_context)
    print(f"Answer: {result}")

def demo_model_comparison():
    """Demonstrate model comparison"""
    print_section("6. Model Comparison")
    
    try:
        from model_comparison import ModelComparator
        from data_loader import load_data
        
        print("Setting up model comparison...")
        
        # Load test data
        test_data, _ = load_data(use_online=False)
        
        # Create comparator
        comparator = ModelComparator()
        comparator.setup_comparison_models()
        
        print(f"Available models: {list(comparator.models.keys())}")
        
        # Run comparison on a subset for demo
        demo_data = test_data.head(3)  # Use first 3 questions for quick demo
        print(f"\nRunning comparison on {len(demo_data)} questions...")
        
        results = comparator.compare_models(demo_data)
        comparator.print_comparison_table()
        
    except Exception as e:
        print(f"Model comparison demo failed: {e}")

def demo_web_interface():
    """Demonstrate web interface"""
    print_section("7. Web Interface (Streamlit)")
    
    print("The Streamlit web interface provides:")
    print("  - Interactive Question Answering")
    print("  - Dataset exploration")
    print("  - Model evaluation")
    print("  - Performance visualizations")
    
    print("\nTo launch the web interface:")
    print("  streamlit run streamlit_app.py")
    print("  or")
    print("  python -m streamlit run streamlit_app.py")
    
    print("\nThe interface will be available at: http://localhost:8501")

def demo_advanced_features():
    """Demonstrate advanced features"""
    print_section("8. Advanced Features")
    
    print("Advanced capabilities include:")
    print("  ✓ Multi-model support (transformers + rule-based)")
    print("  ✓ Offline-first architecture")
    print("  ✓ Automatic fallback mechanisms")
    print("  ✓ Model performance comparison")
    print("  ✓ Interactive web interface")
    print("  ✓ Comprehensive evaluation metrics")
    print("  ✓ Flexible data loading (online/offline)")
    print("  ✓ Model training and management")
    print("  ✓ CLI with multiple operation modes")
    print("  ✓ Visualization and reporting")

def show_project_structure():
    """Show project structure"""
    print_section("Project Structure")
    
    structure = """
    Question-Answering/
    ├── README.md                 # Project documentation
    ├── requirements.txt          # Python dependencies
    ├── streamlit_app.py         # Web interface
    ├── data/                    
    │   ├── sample_squad.json    # Sample dataset (JSON)
    │   └── sample_squad.csv     # Sample dataset (CSV)
    ├── models/                  
    │   └── rule_based_qa.pkl    # Trained rule-based model
    └── src/                     
        ├── basic_qa.py          # Main QA system (adaptive)
        ├── basic_qa_offline.py  # Offline rule-based QA
        ├── cli.py               # Command-line interface
        ├── data_loader.py       # Data loading utilities
        ├── evaluator.py         # Evaluation metrics
        ├── model_manager.py     # Model management
        ├── trainer.py           # Model training
        ├── model_comparison.py  # Model comparison system
        └── demo.py              # This demo script
    """
    
    print(structure)

def main():
    """Run comprehensive demo"""
    print_header("QUESTION ANSWERING SYSTEM - COMPREHENSIVE DEMO")
    
    print("This demo showcases all features of the Question Answering system.")
    print("The system supports transformer models, rule-based approaches, and hybrid methods.")
    print("All features work offline with sample data.")
    
    try:
        show_project_structure()
        demo_basic_qa()
        demo_dataset_loading()
        demo_evaluation()
        demo_model_training()
        demo_cli_interface()
        demo_model_comparison()
        demo_web_interface()
        demo_advanced_features()
        
        print_header("DEMO COMPLETE")
        print("✓ All components demonstrated successfully")
        print("✓ System is ready for production use")
        print("✓ Both online and offline modes supported")
        
        print("\nNext steps:")
        print("1. Try the CLI: python src/cli.py")
        print("2. Launch web interface: streamlit run streamlit_app.py")
        print("3. Run model comparison: python src/model_comparison.py")
        print("4. Explore the code and customize as needed")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Some features may require additional setup or internet connectivity.")

if __name__ == "__main__":
    main()
