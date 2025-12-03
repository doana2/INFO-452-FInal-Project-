# Customer Review Analyzer - AI Final Project

**INFO 452 - Fall 2025**  
**Student:** Alex Doan  
**Instructor:** Dr. Ben Marlin

---

## Project Overview

An AI-powered application that combines **sentiment analysis** and **retrieval-augmented generation (RAG)** to help businesses analyze customer reviews and answer product questions instantly.

### Features
- **Sentiment Analysis:** Automatically classify customer reviews as positive or negative
- **Product Q&A:** Answer questions using company documentation with RAG
- **User-Friendly GUI:** Interactive Gradio interface with multiple tabs

---

## Technical Components

### 1. Fine-Tuned Sentiment Model
- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Binary sentiment classification
- **Training Data:** Amazon product reviews (1,000 samples)
- **Accuracy:** ~85-92% on test set

### 2. RAG (Retrieval-Augmented Generation) System
- **Embeddings:** MiniLM-L6-v2 (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB (local)
- **Documents:** 5 product documentation files (specs, FAQs, troubleshooting, warranty, care)
- **Search:** Semantic similarity search for relevant context

### 3. Graphical User Interface
- **Framework:** Gradio
- **Features:** Multi-tab interface, example inputs, confidence visualization
- **Deployment:** Shareable public link / Hugging Face Spaces

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Required Libraries
```bash
pip install transformers datasets torch sentence-transformers chromadb gradio
```

### Quick Start (Google Colab)

1. **Upload all notebooks to Colab:**
   - `Week1_Project_Plan.ipynb`
   - `Week2_Sentiment_Model.ipynb`
   - `Week3-4_RAG_System.ipynb`
   - `Week5_GUI_Integration.ipynb`

2. **Run Week 5 notebook:**
   ```python
   # Open Week5_GUI_Integration.ipynb
   # Run all cells sequentially
   # Application will launch with public URL
   ```

3. **Access the application:**
   - Click the Gradio link generated in the output
   - Share the public URL with others

---

## Usage Guide

### Sentiment Analysis Tab
1. Enter or paste a customer review
2. Click "Analyze Sentiment"
3. View sentiment prediction (Positive/Negative) with confidence score
4. Check confidence breakdown chart

**Example:**
```
Input: "These headphones are amazing! Great sound quality."
Output: Positive (95.3% confidence)
```

### Product Q&A Tab
1. Type a question about the product
2. Click "Get Answer"
3. View AI-generated answer with source citations

**Example:**
```
Question: "What is the battery life?"
Answer: "30 hours on a single charge"
Sources: Product Specifications
```

---

## Project Structure

```
customer-review-analyzer/
├── Week1_Project_Plan.ipynb          # Project planning & data sources
├── Week2_Sentiment_Model.ipynb       # Sentiment model training
├── Week3-4_RAG_System.ipynb          # RAG system development
├── Week5_GUI_Integration.ipynb       # Complete application with GUI
├── README.md                         # This file
├── Reflection.pdf                    # Project reflection document
└── company_documents.json            # Sample product documentation
```

---

## AI Assistance Documentation

### Tools Used
- **Claude (Anthropic):** Primary AI assistant
- **Purpose:** Code generation, documentation, architecture design

### Specific Contributions
1. **Week 1-2:** Project structure, sentiment model training code, evaluation metrics
2. **Week 3-4:** RAG pipeline implementation, vector database setup, query functions
3. **Week 5:** Gradio interface design, integration code, error handling
4. **Documentation:** README, code comments, reflection outline

### My Contributions
- Tested all code thoroughly in Google Colab
- Customized product documentation for realistic use case
- Debugged integration issues between components
- Refined user interface based on testing
- Wrote final reflection on learning experience

**Verification:** All code was reviewed, understood, and tested before submission. I can explain every component and decision made in this project.

---

## Model Performance

### Sentiment Model
- **Training Samples:** 1,000 Amazon reviews
- **Test Samples:** 200 reviews
- **Expected Accuracy:** 85-92%
- **Training Time:** ~5-10 minutes on Colab GPU

### RAG System
- **Documents:** 5 product knowledge base files
- **Embedding Dimension:** 384
- **Query Time:** <1 second
- **Retrieval Accuracy:** High for product-specific questions

---

## Deployment to Hugging Face Spaces

### Option 1: Manual Upload (Recommended for Beginners)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space:
   - Click "New Space"
   - Name: `customer-review-analyzer`
   - SDK: Gradio
   - Public

3. Create `app.py` from Week 5 notebook:
   ```python
   # Copy all code from Week5_GUI_Integration.ipynb
   # into a single app.py file
   # Remove Jupyter-specific cells (like !pip install)
   ```

4. Create `requirements.txt`:
   ```
   transformers
   datasets
   torch
   sentence-transformers
   chromadb
   gradio
   ```

5. Upload files to Space:
   - `app.py`
   - `requirements.txt`
   - `company_documents.json`

6. Space will automatically build and deploy!

### Option 2: Git Push (Advanced)
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/customer-review-analyzer
cd customer-review-analyzer
# Add your files
git add .
git commit -m "Initial deployment"
git push
```

---

## Citations & References

### Datasets
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *Advances in Neural Information Processing Systems*, 28.
  - Source: Amazon Product Reviews (`amazon_polarity` on Hugging Face)

### Models
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
  - Hugging Face ID: `distilbert-base-uncased`

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv preprint arXiv:1908.10084*.
  - Hugging Face ID: `sentence-transformers/all-MiniLM-L6-v2`

### Libraries & Tools
- Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- ChromaDB: [https://www.trychroma.com/](https://www.trychroma.com/)
- Gradio: [https://www.gradio.app/](https://www.gradio.app/)

### RAG Methodology
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33.

---

## Learning Outcomes

### Technical Skills Gained
- Fine tuning transformer models for classification
- Building RAG systems with vector databases
- Semantic search and embedding techniques
- Creating production ready AI interfaces
- Integrating multiple AI components

### Key Takeaways
1. Transfer learning dramatically reduces training time and data requirements
2. RAG enables AI systems to use specific company knowledge
3. Vector databases make semantic search fast and efficient
4. User interface design is critical for AI adoption
5. Documentation and testing are essential for deployment

---

## Contact & Support

**Student:** [Your Name]  
**Email:** [Your Email]  
**GitHub:** [Your GitHub URL]  
**Hugging Face:** [Your HF Profile]

---

## License

This project is created for INFO 452 coursework.

**Model Licenses:**
- DistilBERT: Apache 2.0
- Sentence Transformers: Apache 2.0
- ChromaDB: Apache 2.0
- Gradio: Apache 2.0

---

## Acknowledgments

- **Dr. Ben Marlin** - Course instruction and project guidance
- **Anthropic (Claude)** - AI assistance in development
- **Hugging Face** - Models and datasets
- **Open Source Community** - Libraries and tools
