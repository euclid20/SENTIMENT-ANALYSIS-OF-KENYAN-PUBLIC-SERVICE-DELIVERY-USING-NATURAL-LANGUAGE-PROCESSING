# Kenyan Public Service Feedback Analyzer 📊

A robust, production-ready Streamlit dashboard powered by a custom Hybrid NLP pipeline to analyze multilingual public sector feedback in Kenya.

## 🏗️ Architecture Overview
This repository documents the entire data lifecycle of the project:
* **Data Extraction:** Automated pagination loops to scrape raw citizen grievances from Twitter/X.
* **Data Preprocessing:** Cleaning, tokenization, and auto-labeling pipelines.
* **Model Training:** Baseline benchmarking (SVM, Naive Bayes) against fine-tuned Transformer models (BERT).
* **Deployment:** A Streamlit web application utilizing `@st.cache_data` for high-speed lexicon lookups and dynamic sentiment overriding.

## ✨ Key Features
* **Dialect Awareness:** Seamlessly handles code-switching between English, Swahili, and Sheng.
* **Sector Contextualization:** Differentiates slang meanings based on the selected sector (Transport, Healthcare, Education).
* **Zero-Retraining Scalability:** The system's vocabulary can be expanded infinitely by updating a local CSV, requiring zero GPU retraining of the core BERT model.

## 📂 Repository Structure
* `1_Data_Extraction/`: Web scraping scripts and raw merged datasets.
* `2_Data_Preprocessing/`: EDA scripts, cleaning pipelines, and labeled ground-truth data.
* `3_Model_Training/`: Google Colab training scripts for the HuggingFace models.
* `app.py`: The core Streamlit application.
* `lexicon.csv`: The dynamic Swahili/Sheng dictionary containing sentiment weights.

## ⚙️ Quick Start
```bash
git clone [https://github.com/YOUR_USERNAME/SENTIMENT-ANALYSIS-OF-KENYAN-PUBLIC-SERVICE-DELIVERY-USING-NATURAL-LANGUAGE-PROCESSING.git](https://github.com/YOUR_USERNAME/SENTIMENT-ANALYSIS-OF-KENYAN-PUBLIC-SERVICE-DELIVERY-USING-NATURAL-LANGUAGE-PROCESSING.git)
cd SENTIMENT-ANALYSIS-OF-KENYAN-PUBLIC-SERVICE-DELIVERY-USING-NATURAL-LANGUAGE-PROCESSING
pip install -r requirements.txt
streamlit run app.py