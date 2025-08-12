"""
Streamlit Web Interface for Question Answering System
A user-friendly web interface for the QA system
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src to path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import local modules
import importlib.util
import model_manager
import data_loader
import evaluator
from basic_qa import simple_qa_system

# Page configuration
st.set_page_config(
    page_title="Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'qa_manager' not in st.session_state:
    with st.spinner("Loading QA system..."):
        try:
            st.session_state.qa_manager = model_manager.create_enhanced_qa_system()
        except:
            st.session_state.qa_manager = simple_qa_system

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🤖 Question Answering System")
    st.markdown("""
    Welcome to the **Enhanced Question Answering System**! This system can answer questions 
    based on provided context using advanced NLP techniques.
    """)
    
    # Sidebar for navigation and model info
    with st.sidebar:
        st.header("🔧 System Information")
        
        # Model info
        model_info = st.session_state.qa_manager.get_model_info()
        st.subheader("Current Model")
        for key, value in model_info.items():
            st.text(f"{key.title()}: {value}")
        
        # Available models
        st.subheader("Available Models")
        available_models = st.session_state.qa_manager.list_available_models()
        for model in available_models:
            st.text(f"• {model}")
        
        st.divider()
        
        # Navigation
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Interactive QA", "📊 Sample Dataset", "📈 Evaluation", "📚 About"]
        )
    
    # Main content based on selected page
    if page == "🏠 Interactive QA":
        interactive_qa_page()
    elif page == "📊 Sample Dataset":
        sample_dataset_page()
    elif page == "📈 Evaluation":
        evaluation_page()
    elif page == "📚 About":
        about_page()

def interactive_qa_page():
    """Interactive Question Answering page"""
    st.header("🏠 Interactive Question Answering")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask a Question")
        
        # Context input
        context = st.text_area(
            "📝 Context (Text Passage):",
            placeholder="Enter the text passage that contains the answer to your question...",
            height=200,
            help="Provide the context/passage that contains information to answer your question."
        )
        
        # Question input
        question = st.text_input(
            "❓ Your Question:",
            placeholder="What would you like to know about the context?",
            help="Ask a question about the context provided above."
        )
        
        # Predict button
        if st.button("🔍 Get Answer", type="primary"):
            if context and question:
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_manager.predict(context, question)
                
                # Display result
                st.success("Answer found!")
                
                # Answer display
                st.subheader("📋 Answer")
                st.info(f"**{result['answer']}**")
                
                # Metrics
                col_conf, col_method = st.columns(2)
                with col_conf:
                    st.metric("Confidence", f"{result['score']:.2%}")
                with col_method:
                    st.metric("Method", result['method'])
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now(),
                    'context': context[:100] + "..." if len(context) > 100 else context,
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'method': result['method']
                })
                
            else:
                st.error("Please provide both context and question!")
    
    with col2:
        st.subheader("💬 Recent Questions")
        
        if st.session_state.chat_history:
            # Show recent questions
            for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {entry['question'][:30]}..."):
                    st.text(f"Context: {entry['context']}")
                    st.text(f"Question: {entry['question']}")
                    st.text(f"Answer: {entry['answer']}")
                    st.text(f"Confidence: {entry['confidence']:.2%}")
                    st.text(f"Method: {entry['method']}")
                    st.text(f"Time: {entry['timestamp'].strftime('%H:%M:%S')}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No questions asked yet. Start by asking a question!")

def sample_dataset_page():
    """Sample dataset exploration page"""
    st.header("📊 Sample Dataset")
    
    # Load sample data
    with st.spinner("Loading sample data..."):
        data, _ = data_loader.load_data(use_online=False)
    
    st.success(f"Loaded {len(data)} sample questions")
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(data))
    with col2:
        st.metric("Topics", data['title'].nunique())
    with col3:
        avg_context_len = data['context'].str.split().str.len().mean()
        st.metric("Avg Context Length", f"{avg_context_len:.0f} words")
    
    # Dataset table
    st.subheader("📋 Dataset Preview")
    st.dataframe(
        data[['id', 'title', 'question', 'answer_text']],
        use_container_width=True
    )
    
    # Question length analysis
    st.subheader("📈 Question Length Distribution")
    question_lengths = data['question'].str.split().str.len()
    fig = px.histogram(
        x=question_lengths,
        nbins=10,
        title="Distribution of Question Lengths (in words)",
        labels={'x': 'Question Length (words)', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Try questions on sample data
    st.subheader("🎯 Test on Sample Data")
    
    selected_idx = st.selectbox(
        "Select a sample question:",
        range(len(data)),
        format_func=lambda x: f"{data.iloc[x]['id']}: {data.iloc[x]['question'][:50]}..."
    )
    
    if st.button("🔍 Answer Selected Question"):
        selected_row = data.iloc[selected_idx]
        
        with st.spinner("Getting answer..."):
            result = st.session_state.qa_manager.predict(
                selected_row['context'], 
                selected_row['question']
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Model Answer")
            st.info(result['answer'])
            st.text(f"Confidence: {result['score']:.2%}")
            st.text(f"Method: {result['method']}")
        
        with col2:
            st.subheader("✅ Expected Answer")
            st.success(selected_row['answer_text'])
            
            # Calculate metrics
            em = evaluator.exact_match_score(result['answer'], selected_row['answer_text'])
            f1 = evaluator.f1_score(result['answer'], selected_row['answer_text'])
            st.text(f"Exact Match: {em:.0%}")
            st.text(f"F1 Score: {f1:.2%}")

def evaluation_page():
    """System evaluation page"""
    st.header("📈 System Evaluation")
    
    # Load test data
    with st.spinner("Loading test data..."):
        test_data, _ = data_loader.load_data(use_online=False)
    
    if st.button("🏃 Run Full Evaluation", type="primary"):
        with st.spinner("Evaluating system on all test examples..."):
            
            # Create evaluation function
            def eval_function(context, question):
                result = st.session_state.qa_manager.predict(context, question)
                return result['answer']
            
            # Run evaluation
            results = evaluator.evaluate_qa_system(eval_function, test_data)
        
        # Display results
        st.success("Evaluation completed!")
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", results['total_questions'])
        with col2:
            st.metric("Exact Match", f"{results['exact_match']:.2%}")
        with col3:
            st.metric("F1 Score", f"{results['f1']:.2%}")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        detailed_df = pd.DataFrame(results['detailed_results'])
        st.dataframe(detailed_df, use_container_width=True)
        
        # Performance visualization
        st.subheader("📊 Performance Visualization")
        
        # F1 scores distribution
        fig = px.histogram(
            x=detailed_df['f1_score'],
            nbins=10,
            title="F1 Score Distribution",
            labels={'x': 'F1 Score', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Exact match vs F1 scatter
        fig2 = px.scatter(
            detailed_df,
            x='exact_match',
            y='f1_score',
            hover_data=['id'],
            title="Exact Match vs F1 Score",
            labels={'exact_match': 'Exact Match', 'f1_score': 'F1 Score'}
        )
        st.plotly_chart(fig2, use_container_width=True)

def about_page():
    """About page with system information"""
    st.header("📚 About the Question Answering System")
    
    st.markdown("""
    ## 🎯 Overview
    This **Question Answering System** demonstrates advanced NLP capabilities using both 
    rule-based and transformer-based approaches. The system can:
    
    - Answer questions based on provided context
    - Work offline with rule-based patterns
    - Support online transformer models when available
    - Provide confidence scores and methodology transparency
    - Evaluate performance with standard QA metrics
    
    ## 🏗️ System Architecture
    """)
    
    # Architecture diagram (text-based)
    st.code("""
    Input (Context + Question)
            ↓
    Model Manager
            ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    │  Trained Rules  │  Transformers   │  Basic Rules    │
    │     (Best)      │   (If Online)   │   (Fallback)    │
    └─────────────────┴─────────────────┴─────────────────┘
            ↓
    Answer + Confidence + Method
    """, language="text")
    
    st.markdown("""
    ## 🔧 Features
    
    ### Core Features
    - **Multi-Model Support**: Rule-based, transformer-based, and hybrid approaches
    - **Offline Capability**: Works without internet connection
    - **Pattern Learning**: Extracts patterns from training data
    - **Model Management**: Automatic model selection and loading
    
    ### Evaluation Metrics
    - **Exact Match (EM)**: Percentage of predictions that match exactly
    - **F1 Score**: Token-level overlap between prediction and ground truth
    - **Confidence Scoring**: Model confidence in predictions
    
    ### Interface Options
    - **Streamlit Web UI**: This interactive web interface
    - **Command Line**: Full-featured CLI with multiple modes
    - **Python API**: Direct integration with model manager
    
    ## 📊 Sample Dataset
    The system includes a curated sample dataset with:
    - **6 high-quality QA examples**
    - **Amazon Rainforest** and **Artificial Intelligence** topics
    - **Proper answer span annotations**
    - **Multiple question types** (percentage, area, definition, etc.)
    
    ## 🚀 Technology Stack
    - **Python 3.12+**
    - **Transformers** (Hugging Face)
    - **PyTorch** 
    - **Pandas** for data handling
    - **Streamlit** for web interface
    - **Regular Expressions** for pattern matching
    
    ## 📈 Performance
    On the sample dataset, the system achieves:
    - **High accuracy** on factual questions
    - **Robust fallback** mechanisms
    - **Fast inference** times
    - **Transparent methodology** reporting
    """)
    
    # System stats
    st.subheader("📊 Current System Stats")
    
    # Get model info
    model_info = st.session_state.qa_manager.get_model_info()
    available_models = st.session_state.qa_manager.list_available_models()
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.info(f"""
        **Current Model**: {model_info.get('type', 'Unknown')}
        **Status**: {model_info.get('status', 'Unknown')}
        **Available Models**: {len(available_models)}
        """)
    
    with stats_col2:
        if st.session_state.chat_history:
            st.success(f"""
            **Questions Asked**: {len(st.session_state.chat_history)}
            **Last Question**: {st.session_state.chat_history[-1]['timestamp'].strftime('%H:%M:%S')}
            **Success Rate**: High
            """)
        else:
            st.warning("No questions asked yet in this session")

if __name__ == "__main__":
    main()
