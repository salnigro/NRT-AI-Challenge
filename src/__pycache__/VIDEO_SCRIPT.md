# Video Presentation Script: NRT-AI Challenge

**Title:** AI-Powered Dual-Reasoning Engine for Substance Abuse Risk
**Estimated Running Time:** ~3-5 Minutes

---

## [0:00 - 0:45] Section 1: The Challenge Problem
**(Visual Idea: Title Slide followed by a split-screen graphic showing social media/text reviews on one side and raw numerical CDC data on the other.)**

**Narrator / Presenter:** 
"Hello everyone, and welcome to our presentation for the NRT-AI Challenge. We tackled the complex issue of substance abuse risk detection. Currently, public health agencies struggle with a critical delay: understanding emerging drug overdoses often requires manually parsing through disconnected channels of unstructured patient complaints and sterile temporal statistical data. 
Our overarching goal was simple: Could we build an explainable, early-warning AI system that concurrently reasons across textual emotional distress and hard demographic statistics to save lives?"

---

## [0:45 - 1:30] Section 2: Dataset and Approach
**(Visual Idea: Screen capture of `data_loader.py` and a brief shot of the raw CSVs. Show a graphic illustrating Table-to-Text conversion.)**

**Narrator / Presenter:** 
"To solve this, we focused explicitly on Innovation Track A: AI Modeling and Reasoning. For our datasets, we maintained strict privacy constraints by relying only on anonymized, population-level public data. 
We fused unstructured patient condition reviews containing emotional signals with deep longitudinal CDC demographic tables—covering age, race, and sex. 

Our unique approach revolved around 'Table-to-Text' generation. Instead of forcing an LLM to guess structured numbers, our pipeline dynamically translated complex statistical arrays into plain English sentences, securely injecting them directly into our training knowledge base alongside the noisy unstructured review data."

---

## [1:30 - 2:30] Section 3: Key Technical Methods
**(Visual Idea: High-level architectural flowchart. Show the RAG process (FAISS -> TinyLlama).)**

**Narrator / Presenter:** 
"Technically, we architected a robust Retrieval-Augmented Generation framework—or RAG. 
When a user asks a nuanced query, a dual-engine kicks in. First, a local semantic embedding model converts the phrase into a dense vector, retrieving the nearest 'K' matching text signals and demographic statistics from our high-speed FAISS vector store. 
Simultaneously, a deterministic time-aware algorithm extracts temporal spikes—like a sudden Year-Over-Year percentage increase in Fentanyl fatalities. 
Both qualitative context and quantitative facts are finally passed to our local analytical brain, `TinyLlama`, governed by specialized LangChain templates to guarantee ethical, interpretable, and hallucination-free outputs."

---

## [2:30 - 3:30] Section 4: Demo / Screenshots of the System
**(Visual Idea: Screen recording of the terminal running `python main.py`.)**

**Narrator / Presenter:** 
"Let’s take a look at the system in action. When we execute our pipeline, you can clearly see the stages initializing. It builds the vector memory using the training set and loads the local models. 
Here, we ask it to analyze: *'Identify any sudden spikes or recurring patterns in synthetic opioids, and correlate them with behavioral patterns found in text.'*
You'll immediately see the terminal retrieving explicit, time-aware percentage changes directly from the CDC module, followed by contextualizing text from patient reviews. The LLM then beautifully synthesizes these into a single, cohesive public health alert. This isn't just an answer—it is an auditable, cited justification."

---

## [3:30 - 4:00] Section 5: Main Results
**(Visual Idea: Bullet points of Results / Conclusion slide.)**

**Narrator / Presenter:** 
"Our main result is a fully functional, local prototype capable of processing massive public health text arrays and outputting highly transparent narratives. The pipeline succeeded in bridging purely linguistic behavioral signals—like relapse mentions—with definitive multi-dimensional CDC demographics. We proved an LLM doesn't have to choose between analyzing human emotion and performing rigid math; it can effectively do both when orchestrated properly."

---

## [4:00 - 4:30] Section 6: Team Contribution Summary
**(Visual Idea: Slide with Team member names and roles.)**

**Narrator / Presenter:** 
"Our team was incredibly collaborative on this build. *(Insert specific member responsibilities here. E.g., Sal led the core architectural design and LLM pipeline construction, while coordinating the Table-to-Text embedding formatting algorithms.)*

Thank you for your time, and we believe tools like this are the next step in proactive public health intelligence."
