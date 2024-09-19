import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv(r'C:/Users/Clyde_glhf/Desktop/Hate Speech ML App/Capstone_Project/Data_Set_Folder/Capstone Dataset.csv')
print(df.head())

# Step 2: Preprocess the data
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Data preprocessing complete.")

# Step 3: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)
print("Model training complete.")

# Step 4: Test the model
y_pred = nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Extract features and labels
# X = df['tweet']
# y = df['class']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical data using CountVectorizer
# vectorizer = CountVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
# nb_classifier = MultinomialNB()

# Train the classifier
# nb_classifier.fit(X_train_vec, y_train)

# Predict the labels for the test set
# y_pred = nb_classifier.predict(X_test_vec)

# Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy * 100:.2f}%")

# Print a detailed classification report
# print(classification_report(y_test, y_pred))

# print(df.head())
# print("Data preprocessing complete.")
# print("Model training complete.")