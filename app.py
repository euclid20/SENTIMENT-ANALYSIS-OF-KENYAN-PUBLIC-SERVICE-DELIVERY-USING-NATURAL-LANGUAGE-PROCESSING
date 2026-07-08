import streamlit as st
import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

st.set_page_config(page_title="Public Service Feedback NLP", layout="centered")
st.title("Kenyan Public Service Feedback Analyzer")

# --- CSV LEXICON LOADER ---
@st.cache_data
def load_lexicon_csv(filepath="lexicon.csv"):
    try:
        df = pd.read_csv(filepath)
        # Clean up headers and text to prevent formatting mismatches
        df.columns = df.columns.str.strip()
        df['Term'] = df['Term'].astype(str).str.strip().str.lower()
        df['Sector'] = df['Sector'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"⚠️ Could not load {filepath}. Please ensure it is in the same folder. Error: {e}")
        return pd.DataFrame(columns=['Term', 'Weight', 'Sector'])

sector = st.selectbox("Select Public Service Sector:", ("Transport", "Education", "Healthcare"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_paths = {
    "Transport": os.path.join(BASE_DIR, "transport_bert_backup"),
    "Education": os.path.join(BASE_DIR, "education_bert_backup"),
    "Healthcare": os.path.join(BASE_DIR, "healthcare_bert_backup")
}

@st.cache_resource
def load_expert_model(sector_choice):
    path = model_paths[sector_choice]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_expert_model(sector)

user_input = st.text_area(f"Enter {sector} Feedback (English/Sheng/Swahili):")

if st.button("Run Analytics"):
    if user_input:
        with st.spinner(f'Analyzing {sector} sentiment...'):
            
            # 1. BERT PREDICTION
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1).squeeze().tolist()
                
            classes = ["Negative", "Neutral", "Positive"]
            prediction_idx = torch.argmax(logits, dim=1).item()
            bert_base_class = classes[prediction_idx]
            confidence_score = max(probabilities) * 100 
            
            # Convert BERT class to a number
            if bert_base_class == "Negative":
                score = -0.8
            elif bert_base_class == "Positive":
                score = 0.8
            else:
                score = 0.0 # Neutral

            # 2. LEXICON OVERRIDE (Now using CSV)
            text_lower = user_input.lower()
            matched_words = []
            
            # Load and filter the CSV based on the selected sector
            lexicon_df = load_lexicon_csv("lexicon.csv")
            
            if not lexicon_df.empty:
                valid_sectors = [sector, "General"]
                filtered_lexicon = lexicon_df[lexicon_df['Sector'].isin(valid_sectors)]
                
                for _, row in filtered_lexicon.iterrows():
                    word = row['Term']
                    weight = float(row['Weight'])
                    
                    if word in text_lower:
                        score += weight
                        matched_words.append(word)
            
            # 3. FINAL DECISION
            if score <= -0.3:
                final_class = "Negative"
            elif score >= 0.3:
                final_class = "Positive"
            else:
                final_class = "Neutral"

            # 4. RENDER UI
            st.markdown("### Analysis Output")
            if final_class == "Negative":
                st.error(f"**Predicted Sentiment:** {final_class}")
            elif final_class == "Positive":
                st.success(f"**Predicted Sentiment:** {final_class}")
            else:
                st.info(f"**Predicted Sentiment:** {final_class}")
                
    else:
        st.warning("Please enter a grievance to analyze.")