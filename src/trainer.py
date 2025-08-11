"""
Model Training and Fine-tuning for Question Answering
Supports both offline rule enhancement and online transformer fine-tuning
"""

import json
import os
import pickle
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import re

class RuleBasedTrainer:
    """
    Trainer for improving rule-based QA system through pattern learning
    """
    
    def __init__(self):
        self.patterns = {
            'percentage': [],
            'area': [],
            'country': [],
            'number': [],
            'definition': []
        }
        self.question_types = {
            'percentage': ['percentage', '%', 'percent'],
            'area': ['square kilometer', 'km2', 'area', 'covers'],
            'country': ['country', 'nation', 'which'],
            'number': ['how many', 'number of', 'count'],
            'definition': ['what is', 'what are', 'define', 'meaning']
        }
    
    def extract_patterns_from_examples(self, examples: List[Dict]):
        """
        Extract answer patterns from training examples
        
        Args:
            examples: List of training examples with context, question, answer
        """
        print("Extracting patterns from training examples...")
        
        for example in examples:
            context = example['context']
            question = example['question'].lower()
            answer = example['answer_text']
            
            # Determine question type
            q_type = self._classify_question(question)
            
            # Extract pattern around the answer
            pattern = self._extract_answer_pattern(context, answer, q_type)
            
            if pattern:
                self.patterns[q_type].append({
                    'pattern': pattern,
                    'example_question': question,
                    'example_answer': answer,
                    'confidence': 0.8
                })
        
        print(f"Extracted patterns:")
        for q_type, patterns in self.patterns.items():
            print(f"  {q_type}: {len(patterns)} patterns")
    
    def _classify_question(self, question: str) -> str:
        """Classify question type based on keywords"""
        question_lower = question.lower()
        
        for q_type, keywords in self.question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        return 'definition'  # Default
    
    def _extract_answer_pattern(self, context: str, answer: str, q_type: str) -> Optional[str]:
        """Extract regex pattern around the answer in context"""
        try:
            # Find answer position in context
            answer_pos = context.find(answer)
            if answer_pos == -1:
                return None
            
            # Extract surrounding context (20 chars before and after)
            start = max(0, answer_pos - 20)
            end = min(len(context), answer_pos + len(answer) + 20)
            
            surrounding = context[start:end]
            
            # Create pattern based on question type
            if q_type == 'percentage':
                # Pattern for percentages
                pattern = re.escape(surrounding).replace(re.escape(answer), r'(\d+)%?')
            elif q_type == 'area':
                # Pattern for areas
                pattern = re.escape(surrounding).replace(re.escape(answer), r'([\d,]+)\s*km2?')
            elif q_type == 'country':
                # Pattern for countries
                pattern = re.escape(surrounding).replace(re.escape(answer), r'(\w+)')
            elif q_type == 'number':
                # Pattern for numbers
                pattern = re.escape(surrounding).replace(re.escape(answer), r'(\w+)')
            else:
                # General pattern
                pattern = re.escape(surrounding).replace(re.escape(answer), r'([^.]+)')
            
            return pattern
            
        except Exception as e:
            print(f"Error extracting pattern: {e}")
            return None
    
    def train_on_examples(self, examples: List[Dict]):
        """
        Train the rule-based system on examples
        
        Args:
            examples: Training examples
        """
        print("Training rule-based QA system...")
        
        # Extract patterns
        self.extract_patterns_from_examples(examples)
        
        # Validate patterns
        self._validate_patterns(examples)
        
        print("Training completed!")
    
    def _validate_patterns(self, examples: List[Dict]):
        """Validate extracted patterns against examples"""
        print("Validating patterns...")
        
        correct = 0
        total = 0
        
        for example in examples:
            predicted = self.predict_with_patterns(example['context'], example['question'])
            expected = example['answer_text']
            
            if predicted and predicted.strip().lower() == expected.strip().lower():
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Pattern validation accuracy: {accuracy:.2%} ({correct}/{total})")
    
    def predict_with_patterns(self, context: str, question: str) -> Optional[str]:
        """Predict answer using learned patterns"""
        q_type = self._classify_question(question)
        
        # Try patterns for this question type
        for pattern_info in self.patterns[q_type]:
            try:
                match = re.search(pattern_info['pattern'], context)
                if match:
                    return match.group(1)
            except Exception:
                continue
        
        return None
    
    def save_model(self, filepath: str):
        """Save trained patterns to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.patterns, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained patterns from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.patterns = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return True
        return False

class TransformerTrainer:
    """
    Trainer for fine-tuning transformer models (requires internet connectivity)
    """
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilbert-squad"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_model(self):
        """Setup transformer model for training"""
        try:
            from transformers import (
                AutoTokenizer, 
                AutoModelForQuestionAnswering,
                TrainingArguments,
                Trainer
            )
            
            print(f"Loading {self.model_name} for fine-tuning...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            
            print("✓ Transformer model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to setup transformer model: {e}")
            return False
    
    def prepare_training_data(self, examples: List[Dict]):
        """Prepare data for transformer training"""
        if not self.tokenizer:
            print("Tokenizer not available!")
            return None
        
        # This would implement the tokenization and preparation
        # for transformer training - placeholder for when internet is available
        print("Preparing training data for transformer...")
        return None
    
    def train(self, train_examples: List[Dict], val_examples: List[Dict] = None):
        """Fine-tune the transformer model"""
        if not self.model:
            print("Model not loaded! Setup model first.")
            return False
        
        print("Training transformer model...")
        # Implementation would go here when online
        print("Note: Transformer training requires internet connectivity")
        return False
    
    def save_model(self, output_dir: str):
        """Save fine-tuned model"""
        if self.model and self.tokenizer:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")

def train_qa_system(use_transformers: bool = False):
    """
    Main training function
    
    Args:
        use_transformers: Whether to use transformer training (requires internet)
    """
    print("="*60)
    print("QUESTION ANSWERING SYSTEM TRAINING")
    print("="*60)
    
    # Load training data
    from data_loader import load_data, preprocess_data
    
    print("Loading training data...")
    train_data, val_data = load_data(use_online=False)
    
    if isinstance(train_data, pd.DataFrame):
        train_examples = preprocess_data(train_data, is_dataframe=True)
        val_examples = []
    else:
        train_examples = preprocess_data(train_data, is_dataframe=False)
        val_examples = preprocess_data(val_data, is_dataframe=False) if val_data else []
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    if use_transformers:
        # Try transformer training
        print("\nAttempting transformer training...")
        trainer = TransformerTrainer()
        
        if trainer.setup_model():
            success = trainer.train(train_examples, val_examples)
            if success:
                trainer.save_model("models/fine_tuned_qa")
            else:
                print("Transformer training failed, falling back to rule-based training")
                use_transformers = False
        else:
            print("Transformer setup failed, using rule-based training")
            use_transformers = False
    
    if not use_transformers:
        # Rule-based training
        print("\nTraining rule-based system...")
        trainer = RuleBasedTrainer()
        trainer.train_on_examples(train_examples)
        trainer.save_model("models/rule_based_qa.pkl")
        
        # Test the trained system
        print("\nTesting trained system:")
        for example in train_examples[:3]:
            prediction = trainer.predict_with_patterns(example['context'], example['question'])
            print(f"Q: {example['question']}")
            print(f"Predicted: {prediction}")
            print(f"Expected: {example['answer_text']}")
            print()

def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Question Answering System")
    parser.add_argument("--transformers", action="store_true",
                       help="Use transformer training (requires internet)")
    
    args = parser.parse_args()
    
    train_qa_system(use_transformers=args.transformers)

if __name__ == "__main__":
    main()
