import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from temporal_analysis import get_drug_trends

def build_llm_pipeline(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Loading local LLM ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use device_map="auto" if GPU is available, else CPU
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)

def analyze_risk_signals(query, llm, vectorstore):
    print(f"Analyzing query: {query}")
    
    # 1. Retrieve unstructured text evidence
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([f"- {d.page_content} (Condition: {d.metadata.get('condition', 'Unknown')})" for d in docs])
    
    # 2. Retrieve quantitative temporal trend evidence
    print("\n[Querying CDC Temporal Database...]")
    stats_context = get_drug_trends(query)
    
    print("\n[Retrieved Unstructured Context]")
    print(context)
    print("\n[Retrieved Temporal Statistics]")
    print(stats_context)
    
    # 3. Hybrid Explainability Prompt
    prompt_template = """<|system|>
You are an expert public health AI analyst. Your task is to extract holistic insights regarding substance abuse risks.
You are provided with unstructured patient reviews, qualitative textual CDC demographic statistics, and numerical CDC overdose statistics.
Synthesize all sources to provide a clear, interpretable rationale answering the user's query.
</s>
<|user|>
Text Context (Patient Reviews & CDC Demographics):
{context}

Temporal Statistical Trends (National CDC Data):
{stats_context}

Query: {query}
</s>
<|assistant|>
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "stats_context", "query"])
    
    chain = prompt | llm
    
    print("\n[LLM Hybrid Reasoning Output]")
    response = chain.invoke({"context": context, "stats_context": stats_context, "query": query})
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
        
    print(response)
    return response
