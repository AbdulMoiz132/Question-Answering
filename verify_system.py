"""
Quick verification script to test all components
"""

import sys
import os
sys.path.append('src')

def test_basic_qa():
    """Test basic QA functionality"""
    print("Testing Basic QA...")
    from basic_qa import simple_qa_system
    
    context = "The Amazon rainforest covers 60% of Brazil and produces 20% of oxygen."
    question = "What percentage of Brazil is Amazon rainforest?"
    
    result = simple_qa_system(context, question)
    print(f"✓ QA System: {result}")
    return result is not None

def test_data_loading():
    """Test data loading"""
    print("Testing Data Loading...")
    from data_loader import load_data
    
    data, _ = load_data(use_online=False)
    print(f"✓ Data loaded: {len(data)} samples")
    return len(data) > 0

def test_evaluation():
    """Test evaluation system"""
    print("Testing Evaluation...")
    from evaluator import exact_match_score, f1_score, evaluate_qa_system
    
    # Test metrics
    em = exact_match_score("60%", "60%")
    f1 = f1_score("60%", "60%")
    print(f"✓ Metrics: EM={em}, F1={f1}")
    return em == 1.0 and f1 == 1.0

def test_model_comparison():
    """Test model comparison"""
    print("Testing Model Comparison...")
    from model_comparison import ModelComparator
    
    comparator = ModelComparator()
    comparator.setup_comparison_models()
    print(f"✓ Models setup: {len(comparator.models)} models")
    return len(comparator.models) > 0

def test_cli():
    """Test CLI components"""
    print("Testing CLI...")
    from cli import QuestionAnsweringCLI
    
    cli = QuestionAnsweringCLI()
    result = cli.answer_question("What is 60%?", "The answer is 60%.")
    print(f"✓ CLI: {result}")
    return result is not None

def main():
    """Run all tests"""
    print("="*50)
    print("COMPONENT VERIFICATION")
    print("="*50)
    
    tests = [
        test_basic_qa,
        test_data_loading,
        test_evaluation,
        test_model_comparison,
        test_cli
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(False)
            print()
    
    print("="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL COMPONENTS WORKING!")
    else:
        print("⚠️  Some components need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
