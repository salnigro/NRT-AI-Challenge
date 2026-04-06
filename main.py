import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import clean_data
from embeddings import build_vector_store, load_vector_store
from reasoning_engine import build_llm_pipeline, analyze_risk_signals

def main():
    print("="*60)
    print("AI for Substance Abuse Risk Detection - Hybrid Track A Pipeline")
    print("="*60)
    print("\nThis pipeline uses local NLP models and structured CDC querying for Risk Signal Extraction.")
    
    if not os.path.exists("data/cleaned_train_reviews.csv") or not os.path.exists("data/cleaned_test_reviews.csv"):
        print("\n[Stage 1] Data Preprocessing")
        print("Cleaned datasets not found. Running preprocessing on train and test data...")
        clean_data()
    else:
        print("\n[Stage 1] Data Preprocessing: Cleaned data already exists (found cleaned train and test datasets).")
        
    if not os.path.exists("data/faiss_index"):
        print("\n[Stage 2] Vector embedding index not found. Building FAISS index...")
        build_vector_store()
    else:
        print("\n[Stage 2] Vector store already exists in data/faiss_index.")
        
    print("\n[Stage 3] Initializing Dual Reasoning Engine")
    try:
        vstore = load_vector_store()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please ensure Stage 2 completed successfully.")
        return
        
    llm = build_llm_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    print("\n[Stage 4] Running Hybrid Explainable AI Analysis")
    queries = [
        "What are the severe risks and recent overdose trends associated with heroin?",
        "Are there instances of emotional distress related to cocaine or synthetic opioids, and are overdoses strictly monitored?",
        "How has the trend for methadone overdoses evolved recently, and what side effects or relapse signals are mentioned in patient reviews?",
        "Identify any sudden spikes or recurring patterns in synthetic opioids, and correlate them with behavioral patterns found in text.",
        "What are the early-warning indicators for overdose based on the textual signals of distress and the temporal data for general drug overdose deaths?"
    ]
    
    for q in queries:
        print("\n" + "~"*50)
        analyze_risk_signals(q, llm, vstore)
        print("~"*50)

if __name__ == "__main__":
    main()
