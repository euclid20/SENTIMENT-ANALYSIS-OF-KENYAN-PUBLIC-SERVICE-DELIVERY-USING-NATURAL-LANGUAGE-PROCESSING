import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the ground-truth labeled transport data
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Healthcare_Labeled.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['Cleaned_Text'])

# 2. Train-Test Stratified Split
X = df['Cleaned_Text']
y = df['Sentiment_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Executing SVM Intermediate Experiment...")

# 3. Vectorization Layer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Support Vector Machine Classifier
# Using a linear kernel as an intermediate step before hyperparameter sweeps
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# 5. Inference
y_pred = svm_classifier.predict(X_test_tfidf)

# 6. Performance Printout
print("==================================================")
print("       SUPPORT VECTOR MACHINE CLASSIFICATION REPORT")
print("==================================================")
print(classification_report(y_test, y_pred, digits=4))
print("==================================================")

# 7. Save Confusion Matrix Plot
output_dir = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Healthcare dataset\Model_Results"
os.makedirs(output_dir, exist_ok=True)

cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["Negative", "Neutral", "Positive"], 
            yticklabels=["Negative", "Neutral", "Positive"])

plt.title('Intermediate SVM Confusion Matrix', fontweight='bold', pad=15)
plt.ylabel('Actual Ground-Truth Sentiment')
plt.xlabel('Predicted Model Sentiment')
plt.tight_layout()

cm_filename = os.path.join(output_dir, 'SVM_Confusion_Matrix.png')
plt.savefig(cm_filename, dpi=300)
plt.close()

print(f"\n✅ SVM Experiment complete. Plot generated at:\n{cm_filename}")