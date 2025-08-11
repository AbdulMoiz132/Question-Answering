"""
Evaluation metrics for Question Answering
Includes Exact Match and F1 Score calculations
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd

def normalize_answer(s: str) -> str:
    """
    Normalize answer text for evaluation
    - Remove articles (a, an, the)
    - Remove punctuation
    - Fix whitespace
    - Convert to lowercase
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate exact match score between prediction and ground truth
    
    Args:
        prediction (str): Predicted answer
        ground_truth (str): Ground truth answer
    
    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score between prediction and ground truth
    
    Args:
        prediction (str): Predicted answer
        ground_truth (str): Ground truth answer
    
    Returns:
        float: F1 score (0.0 to 1.0)
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty predictions
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    
    # Calculate token overlap
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # Calculate precision and recall
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_predictions(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
    """
    Evaluate a list of predictions against ground truths
    
    Args:
        predictions: List of prediction dicts with 'id' and 'prediction' keys
        ground_truths: List of ground truth dicts with 'id' and 'answer' keys
    
    Returns:
        dict: Evaluation results with EM and F1 scores
    """
    # Create lookup for ground truths
    gt_dict = {gt['id']: gt['answer'] for gt in ground_truths}
    
    exact_matches = []
    f1_scores = []
    detailed_results = []
    
    for pred in predictions:
        pred_id = pred['id']
        prediction_text = pred['prediction']
        
        if pred_id in gt_dict:
            ground_truth_text = gt_dict[pred_id]
            
            # Calculate metrics
            em = exact_match_score(prediction_text, ground_truth_text)
            f1 = f1_score(prediction_text, ground_truth_text)
            
            exact_matches.append(em)
            f1_scores.append(f1)
            
            detailed_results.append({
                'id': pred_id,
                'prediction': prediction_text,
                'ground_truth': ground_truth_text,
                'exact_match': em,
                'f1_score': f1
            })
    
    # Calculate averages
    avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return {
        'exact_match': avg_em,
        'f1': avg_f1,
        'total_questions': len(predictions),
        'detailed_results': detailed_results
    }

def evaluate_qa_system(qa_function, test_data: pd.DataFrame) -> Dict:
    """
    Evaluate a QA system on test data
    
    Args:
        qa_function: Function that takes (context, question) and returns answer
        test_data: DataFrame with columns: id, context, question, answer_text
    
    Returns:
        dict: Evaluation results
    """
    predictions = []
    ground_truths = []
    
    print("Evaluating QA system...")
    print("-" * 30)
    
    for _, row in test_data.iterrows():
        # Get prediction from QA system
        try:
            result = qa_function(row['context'], row['question'])
            if isinstance(result, dict) and 'answer' in result:
                prediction = result['answer']
            elif isinstance(result, str):
                prediction = result
            else:
                prediction = str(result)
        except Exception as e:
            print(f"Error predicting for {row['id']}: {e}")
            prediction = "Error"
        
        predictions.append({
            'id': row['id'],
            'prediction': prediction
        })
        
        ground_truths.append({
            'id': row['id'],
            'answer': row['answer_text']
        })
        
        # Print progress
        print(f"Q: {row['question']}")
        print(f"Predicted: {prediction}")
        print(f"Actual: {row['answer_text']}")
        print()
    
    # Evaluate predictions
    results = evaluate_predictions(predictions, ground_truths)
    return results

def print_evaluation_results(results: Dict):
    """
    Print evaluation results in a nice format
    """
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Exact Match (EM): {results['exact_match']:.3f} ({results['exact_match']*100:.1f}%)")
    print(f"F1 Score: {results['f1']:.3f} ({results['f1']*100:.1f}%)")
    print()
    
    print("Detailed Results:")
    print("-" * 40)
    for result in results['detailed_results']:
        print(f"ID: {result['id']}")
        print(f"  Prediction: '{result['prediction']}'")
        print(f"  Ground Truth: '{result['ground_truth']}'")
        print(f"  EM: {result['exact_match']:.1f}, F1: {result['f1']:.3f}")
        print()
    print("="*60)

def main():
    """
    Demo evaluation with sample data
    """
    # Load sample data
    from data_loader import load_data, preprocess_data
    
    print("Loading test data...")
    test_data, _ = load_data(use_online=False)
    
    # Simple demo QA function (rule-based from our offline system)
    def demo_qa_function(context, question):
        """Demo QA function for testing"""
        from basic_qa_offline import simple_qa_system
        result = simple_qa_system(context, question)
        return result['answer']
    
    # Evaluate the system
    results = evaluate_qa_system(demo_qa_function, test_data)
    
    # Print results
    print_evaluation_results(results)

if __name__ == "__main__":
    main()
