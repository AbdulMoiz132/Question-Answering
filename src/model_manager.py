"""
Model Manager for Question Answering System
Handles loading and inference with different model types
"""

import os
import pickle
from typing import Dict, List, Optional, Union
import json

class ModelManager:
    """
    Manages different types of QA models and provides unified interface
    """
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.model_type = None
    
    def load_rule_based_model(self, model_path: str) -> bool:
        """Load rule-based model from pickle file"""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    patterns = pickle.load(f)
                
                # Import the RuleBasedTrainer for inference
                from trainer import RuleBasedTrainer
                model = RuleBasedTrainer()
                model.patterns = patterns
                
                self.models['rule_based'] = model
                self.current_model = model
                self.model_type = 'rule_based'
                
                print(f"✓ Rule-based model loaded from {model_path}")
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading rule-based model: {e}")
            return False
    
    def load_transformer_model(self, model_path: str) -> bool:
        """Load transformer model (requires internet for libraries)"""
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
            
            print(f"Loading transformer model from {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            
            qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer
            )
            
            self.models['transformer'] = {
                'tokenizer': tokenizer,
                'model': model,
                'pipeline': qa_pipeline
            }
            self.current_model = qa_pipeline
            self.model_type = 'transformer'
            
            print(f"✓ Transformer model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            return False
    
    def load_default_offline_model(self) -> bool:
        """Load default offline rule-based system"""
        try:
            from basic_qa_offline import simple_qa_system
            
            self.models['offline'] = simple_qa_system
            self.current_model = simple_qa_system
            self.model_type = 'offline'
            
            print("✓ Default offline model loaded")
            return True
            
        except Exception as e:
            print(f"Error loading offline model: {e}")
            return False
    
    def predict(self, context: str, question: str) -> Dict:
        """
        Make prediction using current model
        
        Args:
            context: Context passage
            question: Question to answer
            
        Returns:
            dict: Prediction result with answer, score, and method
        """
        if not self.current_model:
            return {
                'answer': 'No model loaded',
                'score': 0.0,
                'method': 'none'
            }
        
        try:
            if self.model_type == 'transformer':
                result = self.current_model(question=question, context=context)
                return {
                    'answer': result['answer'],
                    'score': result['score'],
                    'method': 'transformer'
                }
            
            elif self.model_type == 'rule_based':
                predicted = self.current_model.predict_with_patterns(context, question)
                if predicted:
                    return {
                        'answer': predicted,
                        'score': 0.8,
                        'method': 'rule_based_trained'
                    }
                else:
                    # Fallback to offline system
                    from basic_qa_offline import simple_qa_system
                    return simple_qa_system(context, question)
            
            elif self.model_type == 'offline':
                return self.current_model(context, question)
            
            else:
                return {
                    'answer': 'Unknown model type',
                    'score': 0.0,
                    'method': 'error'
                }
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'answer': f'Error: {e}',
                'score': 0.0,
                'method': 'error'
            }
    
    def list_available_models(self) -> List[str]:
        """List available model files"""
        models_dir = "models"
        available = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pkl'):
                    available.append(f"rule_based: {file}")
                elif os.path.isdir(os.path.join(models_dir, file)):
                    # Check if it's a transformer model directory
                    if any(f.startswith('config.json') or f.startswith('pytorch_model') 
                           for f in os.listdir(os.path.join(models_dir, file))):
                        available.append(f"transformer: {file}")
        
        available.append("offline: default rule-based system")
        return available
    
    def get_model_info(self) -> Dict:
        """Get information about current model"""
        if not self.current_model:
            return {'status': 'No model loaded'}
        
        info = {
            'type': self.model_type,
            'status': 'loaded'
        }
        
        if self.model_type == 'rule_based':
            patterns_count = sum(len(patterns) for patterns in self.current_model.patterns.values())
            info['patterns'] = patterns_count
        
        return info

def create_enhanced_qa_system():
    """
    Create an enhanced QA system that tries to load the best available model
    """
    manager = ModelManager()
    
    print("Setting up enhanced QA system...")
    
    # Try to load models in order of preference
    model_loaded = False
    
    # 1. Try trained rule-based model
    if os.path.exists("models/rule_based_qa.pkl"):
        if manager.load_rule_based_model("models/rule_based_qa.pkl"):
            model_loaded = True
    
    # 2. Try transformer model (if available and internet works)
    if not model_loaded and os.path.exists("models/fine_tuned_qa"):
        if manager.load_transformer_model("models/fine_tuned_qa"):
            model_loaded = True
    
    # 3. Fallback to default offline model
    if not model_loaded:
        manager.load_default_offline_model()
    
    return manager

def main():
    """Demo the model manager"""
    print("="*60)
    print("MODEL MANAGER DEMO")
    print("="*60)
    
    # Create model manager
    manager = create_enhanced_qa_system()
    
    # Show available models
    print(f"\nAvailable models:")
    for model in manager.list_available_models():
        print(f"  - {model}")
    
    # Show current model info
    print(f"\nCurrent model info:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test predictions
    print(f"\nTesting predictions:")
    test_cases = [
        ("Brazil has 60% of the Amazon rainforest.", "What percentage does Brazil have?"),
        ("The Amazon basin covers 7,000,000 km2.", "How large is the Amazon basin?"),
    ]
    
    for context, question in test_cases:
        result = manager.predict(context, question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['score']:.2f}")
        print(f"Method: {result['method']}")

if __name__ == "__main__":
    main()
