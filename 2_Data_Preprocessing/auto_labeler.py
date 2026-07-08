import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (only needs to run once)
# nltk.download('vader_lexicon')

# 1. Load the preprocessed data
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Healthcare_Preprocessed.csv"
df = pd.read_csv(file_path)

# Drop any accidental NaNs
df = df.dropna(subset=['Cleaned_Text'])

# 2. Initialize VADER
sia = SentimentIntensityAnalyzer()

# 3. Inject Custom Kenyan Context into VADER
# Standard English VADER thinks "jam" is a tasty fruit spread (neutral/positive). 
# We must force it to understand Kenyan transport realities.
custom_lexicon = {
    # --- Universal Sentiment Markers (Preserved) ---
    'mbaya': -2.0,   # Bad
    'safi': 2.0,     # Clean/Good
    'poa': 2.0,      # Cool/Good
    'fiti': 2.0,     # Fit/Good
    'asante': 2.0,   # Thank you
    'kudos': 2.0,    # Congratulations
    'hongera': 2.0,  # Congratulations
    'shida': -2.0,   # Problem
    'wizi': -3.0,    # Theft/Corruption/Graft
    
    # --- Sector-Specific Healthcare Terms ---
    'mgomo': -2.5,     # Strike (e.g., doctors/nurses strike)
    'ugonjwa': -2.0,   # Sickness/Disease
    'maumivu': -2.0,   # Pain/Suffering
    'foleni': -1.5,    # Extremely long queues/waiting times
    'mauti': -3.0,     # Death/Mortality
    'kupona': 2.5,     # Recovery/Healing
    'nafuu': 2.0,      # Relief/Affordable medical bill
    'stockout': -2.5,  # Lack of essential drugs in facilities
    'bila': -1.5,      # "Without" (frequently used as "bila dawa" - without medicine)
    'crisis': -2.5,    # Hospital management/SHIF transition hitches
    'negligence': -3.0 # Medical malpractice or poor care
}
sia.lexicon.update(custom_lexicon)

# 4. Function to assign sentiment labels based on compound score
def get_sentiment(text):
    if not isinstance(text, str):
        return "Neutral"
    
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Thresholds: Anything between -0.05 and 0.05 is neutral
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# 5. Apply the labeling
print("Auto-labeling tweets... injecting custom Kenyan context...")
df['Sentiment_Score'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['Sentiment_Label'] = df['Cleaned_Text'].apply(get_sentiment)

# 6. Print the distribution of our new labels
print("\n--- Sentiment Label Distribution ---")
print(df['Sentiment_Label'].value_counts())

# View a sample to verify it worked
print("\n--- Sample Labeled Tweets ---")
print(df[['Cleaned_Text', 'Sentiment_Score', 'Sentiment_Label']].head(10))

# 7. Save the labeled dataset
output_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Healthcare_Labeled.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Ground-truth labeled data safely saved to:\n{output_path}")