import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the ground-truth labeled transport data
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Healthcare_Labeled.csv"
df = pd.read_csv(file_path)

# Ensure no empty text blocks disrupt vectorization
df = df.dropna(subset=['Cleaned_Text'])

# 2. Extract Features and Target Variable
X = df['Cleaned_Text']
y = df['Sentiment_Label']

# 3. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains perfect label ratios across sets
)

print(f"Executing Baseline Experiment...")
print(f"Total Training Samples: {len(X_train)}")
print(f"Total Testing Samples:  {len(X_test)}\n")

# 4. Feature Representation Layer (TF-IDF Vectorization)
# Restricting features to top 5000 n-grams to save computational overhead
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Model Training (Multinomial Naive Bayes)
nb_classifier = MultinomialNB(alpha=1.0)  # Standard Laplace Smoothing
nb_classifier.fit(X_train_tfidf, y_train)

# 6. Predictive Inference
y_pred = nb_classifier.predict(X_test_tfidf)

# 7. Print Classification Metrics to Terminal
print("==================================================")
print("     NAIVE BAYES BASELINE CLASSIFICATION REPORT   ")
print("==================================================")
print(classification_report(y_test, y_pred, digits=4))
print("==================================================")

# 8. Render and Save Confusion Matrix Plot
output_dir = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Model_Results"
os.makedirs(output_dir, exist_ok=True)

cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=["Negative", "Neutral", "Positive"], 
            yticklabels=["Negative", "Neutral", "Positive"])

plt.title('Baseline Multinomial Naive Bayes Confusion Matrix', fontweight='bold', pad=15)
plt.ylabel('Actual Ground-Truth Sentiment')
plt.xlabel('Predicted Model Sentiment')
plt.tight_layout()

cm_filename = os.path.join(output_dir, 'Naive_Bayes_Confusion_Matrix.png')
plt.savefig(cm_filename, dpi=300)
plt.close()

print(f"\n✅ Experiment complete. Confusion Matrix plot generated at:\n{cm_filename}")