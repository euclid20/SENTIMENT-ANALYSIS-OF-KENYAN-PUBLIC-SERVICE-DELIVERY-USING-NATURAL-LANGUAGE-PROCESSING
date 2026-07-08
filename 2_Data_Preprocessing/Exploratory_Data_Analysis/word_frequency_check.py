import pandas as pd
from collections import Counter

# 1. Load the preprocessed dataset
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Education dataset\Education_Preprocessed.csv"
df = pd.read_csv(file_path)

# 2. Drop any missing values just in case
df = df.dropna(subset=['Cleaned_Text'])

# 3. Combine all tweets into one massive string, then split into individual words
print("Analyzing word frequencies...")
all_words = " ".join(df['Cleaned_Text'].tolist()).split()

# 4. Count the frequency of each word
word_counts = Counter(all_words)

# 5. Extract and print the top 30 most common words
top_30_words = word_counts.most_common(30)

print("\n--- Top 30 Most Frequent Words ---")
print(f"{'Word':<15} | {'Count'}")
print("-" * 25)
for word, count in top_30_words:
    print(f"{word:<15} | {count}")