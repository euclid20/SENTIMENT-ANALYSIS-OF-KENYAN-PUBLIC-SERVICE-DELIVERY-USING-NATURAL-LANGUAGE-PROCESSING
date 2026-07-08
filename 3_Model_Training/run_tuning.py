import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# 1. Load the data
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Education dataset\Education_Labeled.csv"
df = pd.read_csv(file_path).dropna(subset=['Cleaned_Text'])

X = df['Cleaned_Text']
y = df['Sentiment_Label']

# 2. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Vectorization Layer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)

# 4. Define the Hyperparameter Search Space
param_grid = {
    'C': [0.1, 1, 10],                   # Regularization strength
    'kernel': ['linear', 'rbf'],         # Geometric boundary shape
    'class_weight': ['balanced', None]   # Handling dataset imbalance
}

print("Initializing Grid Search Cross-Validation for SVM...")
print("Testing 12 different hyperparameter combinations (5-Fold CV)...")
print("This may take a few minutes. Please wait...\n")

# 5. Execute Grid Search
svm_model = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

grid_search.fit(X_train_tfidf, y_train)

# 6. Output the Optimal Results
print("==================================================")
print("           HYPERPARAMETER TUNING RESULTS          ")
print("==================================================")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validation Macro F1-Score: {grid_search.best_score_:.4f}")
print("==================================================")