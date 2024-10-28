import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# 1. Load and Prepare the Data
# Here, we assume that the dataset is a CSV file with columns 'text' (message) and 'label' (spam or ham)
# Replace 'your_dataset.csv' with the path to your spam detection dataset.
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
print(data.columns)
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})  # Encode labels: 0 = ham, 1 = spam

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# 3. Define the Text Processing and Model Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),  # TF-IDF for text feature extraction
    ('clf', MultinomialNB())  # Naive Bayes classifier
])

# 4. Train the Model
pipeline.fit(X_train, y_train)

# 5. Predict on the Test Set
y_pred = pipeline.predict(X_test)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Optional: Function to Detect and Block Spam Messages
def detect_spam(message):
    prediction = pipeline.predict([message])
    if prediction[0] == 1:
        return "Spam - Message Blocked"
    else:
        return "Ham - Message Allowed"

# Example usage
sample_message = "Congratulations! You've won a free ticket to the Bahamas. Claim now!"
print(detect_spam(sample_message))
