import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NOTE: The first time you run NLTK, you need to download its dictionaries.
# Uncomment the two lines below for the very first run, then you can comment them out again.
# nltk.download('punkt')
# nltk.download('stopwords')

# 1. Load the raw concatenated dataset
file_path = r"C:\Users\User1\Folder1\2_Data_Preprocessing\Raw_Transport_Data.csv"
df = pd.read_csv(file_path)

print(f"Original Dataset Size: {df.shape}")

# 2. Handling Missing Values and Duplicates
df.dropna(subset=['Tweet_Text'], inplace=True)
df.drop_duplicates(subset=['Tweet_ID'], inplace=True)

# 3. Define a custom hybrid stopword list
english_stops = set(stopwords.words('english'))
# Add common Swahili/Sheng filler words that don't carry sentiment
kenyan_stops = {'na', 'ya', 'wa', 'kwa', 'ni', 'za', 'hiyo', 'hii', 'kama', 'tu', 'kila', 'lakini', 'kuna', 'pia', 'ndio', 'sana', 'amp', 'along', 'one', 'us', 'day', 'time'}
all_stops = english_stops.union(kenyan_stops)

def advanced_clean_tweet(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase text to ensure uniformity
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username)
    text = re.sub(r'\@\w+', '', text)
    
    # Remove RT (Retweet) tags
    text = re.sub(r'^rt[\s]+', '', text)
    
    # Remove special characters, numbers, and punctuations (keeping only letters)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in all_stops]
    
    return " ".join(cleaned_tokens)

# 4. Apply the advanced cleaning function
print("Cleaning text data... this might take a moment.")
df['Cleaned_Text'] = df['Tweet_Text'].apply(advanced_clean_tweet)

# 5. Drop rows that became empty after cleaning (e.g., a tweet that was just a link)
df = df[df['Cleaned_Text'].str.strip() != ""]

# 6. Feature Engineering: Add Word Count (useful for Exploratory Data Analysis later)
df['Word_Count'] = df['Cleaned_Text'].apply(lambda x: len(str(x).split()))

print(f"Final Preprocessed Dataset Size: {df.shape}")

# View the first 5 before-and-after rows to verify the transformation
print("\n--- Sample Text Transformation ---")
print(df[['Tweet_Text', 'Cleaned_Text']].head())

# Save the preprocessed output
output_path = r"C:\Users\User1\Folder1\2_Data_Preprocessing\Transport_Preprocessed.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Preprocessed data safely saved to: {output_path}")