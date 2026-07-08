import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

# 1. Setup and Load Data
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Education dataset\Education_Preprocessed.csv"
df = pd.read_csv(file_path)

# Drop any remaining NaNs in Cleaned_Text just to be safe
df = df.dropna(subset=['Cleaned_Text'])

# Ensure Word_Count is numeric
df['Word_Count'] = pd.to_numeric(df['Word_Count'], errors='coerce')

# Set aesthetic style for the plots
sns.set_theme(style="whitegrid")

# Create a directory to save the plots (highly recommended for your thesis document)
output_dir = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Education dataset\EDA_Plots"
os.makedirs(output_dir, exist_ok=True)

print("Generating EDA Visualizations...")

# --- PLOT 1: Tweet Length Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Word_Count'], bins=30, kde=True, color='royalblue')
plt.title('Distribution of Tweet Word Counts (Education Sector)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Words per Tweet', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_Word_Count_Distribution.png'))
plt.close()
print("✅ Saved: Word Count Distribution")

# --- PLOT 2: Top 20 Most Frequent Words ---
all_words = " ".join(df['Cleaned_Text'].tolist()).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)
words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Word', data=words_df, palette='viridis')
plt.title('Top 20 Most Frequent Words (Education Sector)', fontsize=14, fontweight='bold')
plt.xlabel('Frequency Count', fontsize=12)
plt.ylabel('Word', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_Top_20_Words.png'))
plt.close()
print("✅ Saved: Top 20 Words Bar Chart")

# --- PLOT 3: Engagement Metrics Correlation Heatmap ---
# Selecting only the numerical engagement columns + Word_Count
numeric_cols = ['Likes', 'Retweets', 'Replies', 'Quotes', 'Views', 'Word_Count']
# Convert to numeric, forcing errors to NaN, then fill with 0
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Engagement Metrics & Word Count', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_Correlation_Heatmap.png'))
plt.close()
print("✅ Saved: Correlation Heatmap")

# --- PLOT 4: Word Cloud ---
text_for_cloud = " ".join(df['Cleaned_Text'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno', max_words=150).generate(text_for_cloud)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Education Sector Tweets', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_Word_Cloud.png'))
plt.close()
print("✅ Saved: Word Cloud")

print(f"\n🎉 All EDA visualizations have been successfully saved to:\n{output_dir}")