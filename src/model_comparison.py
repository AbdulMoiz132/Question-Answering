"""
Model Comparison System
Compare different QA models and approaches
"""

import pandas as pd
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
import os

class ModelComparator:
    """
    Compare different QA models on the same dataset
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_function, description: str = ""):
        """Add a model to compare"""
        self.models[name] = {
            'function': model_function,
            'description': description
        }
        print(f"Added model: {name}")
    
    def setup_comparison_models(self):
        """Setup different models for comparison"""
        print("Setting up models for comparison...")
        
        # 1. Basic offline rule-based system
        from basic_qa_offline import simple_qa_system
        self.add_model(
            "Basic Rules",
            lambda context, question: simple_qa_system(context, question),
            "Basic rule-based system with hardcoded patterns"
        )
        
        # 2. Enhanced rule-based system (trained)
        try:
            from model_manager import ModelManager
            manager = ModelManager()
            if manager.load_rule_based_model("models/rule_based_qa.pkl"):
                self.add_model(
                    "Trained Rules",
                    lambda context, question: manager.predict(context, question),
                    "Rule-based system with learned patterns"
                )
        except Exception as e:
            print(f"Could not load trained rules: {e}")
        
        # 3. Hybrid system with enhanced patterns
        from basic_qa import simple_qa_system as enhanced_simple_qa
        self.add_model(
            "Enhanced Rules",
            lambda context, question: enhanced_simple_qa(context, question),
            "Enhanced rule-based with flexible patterns"
        )
        
        # 4. Transformer models (if available)
        # Note: These would be loaded when internet is available
        transformer_models = [
            "distilbert-base-cased-distilbert-squad",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2"
        ]
        
        for model_name in transformer_models:
            try:
                # This would work when internet is available
                # For now, we'll create placeholder entries
                self.add_model(
                    f"Transformer-{model_name.split('-')[0].upper()}",
                    self._create_transformer_placeholder(model_name),
                    f"Transformer model: {model_name}"
                )
            except Exception as e:
                print(f"Transformer {model_name} not available: {e}")
    
    def _create_transformer_placeholder(self, model_name: str):
        """Create placeholder for transformer models when offline"""
        def placeholder_function(context, question):
            return {
                'answer': f"[Offline - {model_name} unavailable]",
                'score': 0.0,
                'method': f'transformer-{model_name}',
                'note': 'Requires internet connection'
            }
        return placeholder_function
    
    def compare_models(self, test_data: pd.DataFrame) -> Dict:
        """
        Compare all models on test data
        
        Returns:
            dict: Comparison results
        """
        print("="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\nTesting {model_name}...")
            model_function = model_info['function']
            
            model_results = []
            total_time = 0
            
            for _, row in test_data.iterrows():
                start_time = time.time()
                
                try:
                    result = model_function(row['context'], row['question'])
                    
                    # Normalize result format
                    if isinstance(result, dict):
                        answer = result.get('answer', 'No answer')
                        confidence = result.get('score', 0.0)
                        method = result.get('method', model_name.lower())
                    else:
                        answer = str(result)
                        confidence = 0.5
                        method = model_name.lower()
                    
                    # Calculate metrics
                    from evaluator import exact_match_score, f1_score
                    em = exact_match_score(answer, row['answer_text'])
                    f1 = f1_score(answer, row['answer_text'])
                    
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    
                    model_results.append({
                        'id': row['id'],
                        'question': row['question'],
                        'expected': row['answer_text'],
                        'predicted': answer,
                        'exact_match': em,
                        'f1_score': f1,
                        'confidence': confidence,
                        'inference_time': inference_time
                    })
                    
                except Exception as e:
                    print(f"  Error on {row['id']}: {e}")
                    model_results.append({
                        'id': row['id'],
                        'question': row['question'],
                        'expected': row['answer_text'],
                        'predicted': f"Error: {e}",
                        'exact_match': 0.0,
                        'f1_score': 0.0,
                        'confidence': 0.0,
                        'inference_time': 0.0
                    })
            
            # Calculate aggregate metrics
            total_questions = len(model_results)
            avg_em = sum(r['exact_match'] for r in model_results) / total_questions
            avg_f1 = sum(r['f1_score'] for r in model_results) / total_questions
            avg_confidence = sum(r['confidence'] for r in model_results) / total_questions
            avg_inference_time = total_time / total_questions
            
            results[model_name] = {
                'description': model_info['description'],
                'total_questions': total_questions,
                'exact_match': avg_em,
                'f1_score': avg_f1,
                'avg_confidence': avg_confidence,
                'avg_inference_time': avg_inference_time,
                'total_time': total_time,
                'detailed_results': model_results
            }
            
            print(f"  ✓ EM: {avg_em:.2%}, F1: {avg_f1:.2%}, Time: {avg_inference_time:.3f}s")
        
        self.results = results
        return results
    
    def print_comparison_table(self):
        """Print a formatted comparison table"""
        if not self.results:
            print("No results available. Run compare_models() first.")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create table
        print(f"{'Model':<20} {'EM':<8} {'F1':<8} {'Conf':<8} {'Time(s)':<10} {'Description'}")
        print("-" * 80)
        
        # Sort by F1 score
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        for model_name, results in sorted_results:
            em = results['exact_match']
            f1 = results['f1_score']
            conf = results['avg_confidence']
            time_avg = results['avg_inference_time']
            desc = results['description'][:30] + "..." if len(results['description']) > 30 else results['description']
            
            print(f"{model_name:<20} {em:<8.2%} {f1:<8.2%} {conf:<8.2f} {time_avg:<10.3f} {desc}")
        
        print("-" * 80)
    
    def save_results(self, filepath: str = "model_comparison_results.json"):
        """Save comparison results to file"""
        if not self.results:
            print("No results to save.")
            return
        
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'description': results['description'],
                'metrics': {
                    'exact_match': results['exact_match'],
                    'f1_score': results['f1_score'],
                    'avg_confidence': results['avg_confidence'],
                    'avg_inference_time': results['avg_inference_time'],
                    'total_questions': results['total_questions']
                },
                'detailed_results': results['detailed_results']
            }
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def create_comparison_plots(self):
        """Create visualization plots for comparison"""
        if not self.results:
            print("No results available for plotting.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            models = list(self.results.keys())
            em_scores = [self.results[m]['exact_match'] for m in models]
            f1_scores = [self.results[m]['f1_score'] for m in models]
            inference_times = [self.results[m]['avg_inference_time'] for m in models]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # EM scores
            ax1.bar(models, em_scores)
            ax1.set_title('Exact Match Scores')
            ax1.set_ylabel('EM Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # F1 scores
            ax2.bar(models, f1_scores)
            ax2.set_title('F1 Scores')
            ax2.set_ylabel('F1 Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Inference times
            ax3.bar(models, inference_times)
            ax3.set_title('Average Inference Time')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
            
            # EM vs F1 scatter
            ax4.scatter(em_scores, f1_scores)
            for i, model in enumerate(models):
                ax4.annotate(model, (em_scores[i], f1_scores[i]))
            ax4.set_xlabel('Exact Match')
            ax4.set_ylabel('F1 Score')
            ax4.set_title('EM vs F1 Scores')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Comparison plots saved as 'model_comparison.png'")
            
        except ImportError:
            print("Matplotlib not available for plotting.")

def main():
    """Main comparison function"""
    # Load test data
    from data_loader import load_data
    
    print("Loading test data...")
    test_data, _ = load_data(use_online=False)
    
    # Create comparator
    comparator = ModelComparator()
    comparator.setup_comparison_models()
    
    # Run comparison
    results = comparator.compare_models(test_data)
    
    # Display results
    comparator.print_comparison_table()
    
    # Save results
    comparator.save_results()
    
    # Create plots (if matplotlib available)
    try:
        comparator.create_comparison_plots()
    except Exception as e:
        print(f"Could not create plots: {e}")

if __name__ == "__main__":
    main()
