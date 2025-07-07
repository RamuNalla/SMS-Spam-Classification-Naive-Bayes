import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re # For regex operations

#---------------------------------- STEP-1: DATA LOADING AND EDA -----------------------------------------

df = pd.read_csv("SMSSpamCollection.csv", sep="\t", header = None, names = ["Label", "Message"])    # Since, their is no header, read_csv consider first row as the header. Therefore header = "none" and assigned labels with names array

df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# Define a custom preprocessor for CountVectorizer to handle lowercasing,  punctuation removal, and tokenization.
# CountVectorizer has its own stop word removal, so we'll let it handle that.
def custom_preprocessor(text):
    text = text.lower()                             # Lowercase
    text = re.sub(r'[^\w\s]', '', text)             # Remove punctuation
    return text

X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#---------------------------------- STEP-2: FEATURE EXTRACTION (TEXT VECTORIZATION) -----------------------------------------

vectorizer = CountVectorizer(preprocessor=custom_preprocessor, stop_words='english', min_df=1, max_df=0.9)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)



#---------------------------------- STEP-3: NAIVE BAYES CLASSIFIER AND TRAINING -----------------------------------------

model_sklearn = MultinomialNB(alpha=1.0)     # Initialize Multinomial Naive Bayes classifier. alpha=1.0 applies Laplace smoothing (add-1 smoothing), similar to our scratch implementation.

model_sklearn.fit(X_train_vectorized, y_train)


#---------------------------------- STEP-3: PREDICTION AND EVALUATION -----------------------------------------

y_pred_sklearn = model_sklearn.predict(X_test_vectorized)
y_prob_sklearn = model_sklearn.predict_proba(X_test_vectorized)[:, 1]

# Calculate classification metrics.
accuracy = accuracy_score(y_test, y_pred_sklearn)
precision = precision_score(y_test, y_pred_sklearn)             # default pos_label=1
recall = recall_score(y_test, y_pred_sklearn)                   # default pos_label=1
f1_score_ = f1_score(y_test, y_pred_sklearn)                    # default pos_label=1
conf_matrix = confusion_matrix(y_test, y_pred_sklearn)
class_report = classification_report(y_test, y_pred_sklearn, target_names=['ham', 'spam'])


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score_:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("  [[True Negatives (Ham)  False Positives (Spam)]")
print("   [False Negatives (Ham) True Positives (Spam)]]")
print("\nClassification Report:\n", class_report)


#---------------------------------- STEP-4: VISUALIZATION -----------------------------------------

plt.figure(figsize=(12, 6))

# Plot 1: Confusion Matrix Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"])
plt.title('Confusion Matrix (Scikit-learn)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot 2: Distribution of predicted probabilities for Spam vs Ham
# This helps visualize how well the model separates the classes based on its confidence.
plt.subplot(1, 2, 2)
sns.histplot(y_prob_sklearn[y_test == 0], color='blue', label='Ham (Actual 0)', kde=True, stat='density', alpha=0.7, bins=50)
sns.histplot(y_prob_sklearn[y_test == 1], color='red', label='Spam (Actual 1)', kde=True, stat='density', alpha=0.7, bins=50)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Distribution of Predicted Probabilities (Scikit-learn)')
plt.xlabel('Predicted Probability of Spam')
plt.ylim(0, 10)
plt.ylabel('Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#---------------------------------- STEP-5: INTERPRETATION OF METRICS -----------------------------------------

print(f"Accuracy ({accuracy:.4f}): The overall proportion of correctly classified messages.")
print(f"Precision ({precision:.4f}): Out of all messages predicted as SPAM, {precision*100:.2f}% were actually SPAM.")
print(f"Recall ({recall:.4f}): Out of all actual SPAM messages, the model correctly identified {recall*100:.2f}% of them.")
print(f"F1-Score ({f1_score_:.4f}): A balanced measure of precision and recall.")
print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives ({conf_matrix[0,0]}): Correctly identified Ham messages.")
print(f"  False Positives ({conf_matrix[0,1]}): Ham messages incorrectly classified as Spam (Type I error - 'false alarm').")
print(f"  False Negatives ({conf_matrix[1,0]}): Spam messages incorrectly classified as Ham (Type II error - 'missed spam').")
print(f"  True Positives ({conf_matrix[1,1]}): Correctly identified Spam messages.")
