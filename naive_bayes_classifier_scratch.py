import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------- STEP-1: DATA LOADING AND EDA -----------------------------------------

data = pd.read_csv("SMSSpamCollection.csv", sep="\t", header = None, names = ["Label", "Message"])    # Since, their is no header, read_csv consider first row as the header. Therefore header = "none" and assigned labels with names array

data["Message"] = data["Message"].str.lower()
data["Message"] = data["Message"].str.replace(r'[^\w\s]', '', regex=True)   # replace all character which are not w-word, s-white space with ''-nothing
data["Message"] = data["Message"].str.split()                               # Splits text into words  ##Tokenizes text into individual words

stop_words = {"the", "a", "is", "to", "and", "in", "for", "of", "on"}                                       # removing common "stop words" (like "the", "a", "is") using a predefined list
data["Message"] = data["Message"].apply(lambda words: [word for word in words if word not in stop_words])   # lambda function to remove the stop_words     

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

print(data.head())


#---------------------------------- STEP-2: VOCABULARY BUILDING AND NB PARAMETER COMPUTATION -----------------------------------------

# Building a vocabulary on the entire words present in data_train by removing duplicates
vocabulary = set(word for message in data_train["Message"] for word in message)    # Loops through each message in data_train and loops through each word in each message. Set removes all the duplicates, Vocabulary now contains all unique words

# Calculating probabilities of Ham and Spam
num_spam = (data_train["Label"] == "spam").sum()
num_ham = (data_train["Label"] == "ham").sum()

total_messages = len(data_train)

P_spam = num_spam/total_messages
P_ham = num_ham/total_messages


# Counting occurrences for each word seperately in spam and ham messages 

## Intialize Dictionaries to store word counts. Dictionary where keys are words and values are number of times each word appears
spam_word_counts = {}
ham_word_counts = {}  

spam_messages = data_train[data_train["Label"] == "spam"]["Message"]    #Filter rows where the label is "spam" and keepts its messages
ham_messages = data_train[data_train["Label"] == "ham"]["Message"]

for message in spam_messages:
    for word in message:
        spam_word_counts[word] = spam_word_counts.get(word, 0) + 1   # get function gets the value for a particular key, if word is not present it gives 0, Therefore adding 1 when it gives 0

for message in ham_messages:
    for word in message:
        ham_word_counts[word] = ham_word_counts.get(word, 0) + 1 

#Apply laplace smoothing and computing probabilities

total_words_in_spam = sum(spam_word_counts.values())  # Calculate total word in spam and ham messages
total_words_in_ham = sum(ham_word_counts.values())

vocab_size = len(vocabulary)    # Vocab contains all words without duplication

P_word_given_spam = {}          # Probabiltiies are stored in dictionaries
P_word_given_ham = {}

for word in vocabulary:
    P_word_given_spam[word] = (spam_word_counts.get(word, 0) + 1) / (total_words_in_spam + vocab_size)     # +1 is added to each word count to avoid zero probabilities.
    P_word_given_ham[word] = (ham_word_counts.get(word, 0) + 1) / (total_words_in_ham + vocab_size)


#---------------------------------- STEP-3: NAIVE BAYES CLASSIFIER AND CLASSIFICATION -----------------------------------------

def classify(message):          # Building a Naive Bayes Classifier 

    spam_score = np.log(P_spam)
    ham_score = np.log(P_ham)

    words = [word.lower() for word in message]
    words = [word for word in words if word.isalnum()]          # Preprocess the message by removing punctuation and keeping only alphanumeriv words

    for word in words:
        if word in vocabulary:
            spam_score += np.log(P_word_given_spam.get(word, 1/(total_words_in_spam+vocab_size)))
            ham_score += np.log(P_word_given_ham.get(word, 1/(total_words_in_ham+vocab_size)))

    return "spam" if spam_score > ham_score else "ham"

# Apply classifier to test set
y_pred = data_test["Message"].apply(classify)


# Evaluate performance
accuracy = accuracy_score(data_test["Label"], y_pred)
precision = precision_score(data_test["Label"], y_pred, pos_label="spam")     # We need to label positive class when calculing precision and recall
recall = recall_score(data_test["Label"], y_pred, pos_label="spam")
conf_matrix = confusion_matrix(data_test["Label"], y_pred, labels=["ham", "spam"])
f1_score_ = f1_score(data_test["Label"], y_pred, pos_label="spam")

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score_:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("  [[True Negatives (Ham)  False Positives (Spam)]")
print("   [False Negatives (Ham) True Positives (Spam)]]")

#---------------------------------- STEP-4: VISUALIZATION -----------------------------------------

plt.figure(figsize=(12, 6))

# Plot 1: Confusion Matrix Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"])
plt.title('Confusion Matrix (From Scratch)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot 2: Top N words in Spam vs Ham (simple visualization)
# You might need to adjust this for better visual clarity or choose different features
def get_top_n_words(word_counts, n=10):
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_words[:n]

top_spam_words = get_top_n_words(spam_word_counts, n=10)
top_ham_words = get_top_n_words(ham_word_counts, n=10)

words_spam = [word for word, count in top_spam_words]
counts_spam = [count for word, count in top_spam_words]
words_ham = [word for word, count in top_ham_words]
counts_ham = [count for word, count in top_ham_words]

plt.subplot(1, 2, 2)
plt.barh(words_spam, counts_spam, color='red', alpha=0.7, label='Spam')
plt.barh(words_ham, counts_ham, color='blue', alpha=0.7, left=np.array(counts_spam) if len(counts_spam) == len(counts_ham) else None, label='Ham') # Simple stacking if sizes match
plt.title('Top 10 Most Frequent Words (Training Data)')
plt.xlabel('Word Count')
plt.ylabel('Word')
plt.gca().invert_yaxis() # Puts the highest count at the top
plt.legend()

plt.tight_layout()
plt.show()

#---------------------------------- STEP-5: INTERPRETATION OF METRICS -----------------------------------------

print(f"Accuracy ({accuracy:.4f}): The overall proportion of correctly classified messages (both spam and ham).")
print(f"Precision ({precision:.4f}): Out of all messages predicted as SPAM, {precision*100:.2f}% were actually SPAM")
print(f"Recall ({recall:.4f}): Out of all actual SPAM messages, the model correctly identified {recall*100:.2f}% of them")
print(f"F1-Score ({f1_score_:.4f}): The harmonic mean of Precision and Recall.")
print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives ({conf_matrix[0,0]}): Correctly classified as HAM.")
print(f"  False Positives ({conf_matrix[0,1]}): Incorrectly classified as SPAM (a HAM message was flagged as SPAM). This is a Type I error.")
print(f"  False Negatives ({conf_matrix[1,0]}): Incorrectly classified as HAM (a SPAM message was missed). This is a Type II error.")
print(f"  True Positives ({conf_matrix[1,1]}): Correctly classified as SPAM.")
