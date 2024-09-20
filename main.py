#Gabriel Gallardo Capstone Project 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import RandomOverSampler
import random

# Step 1: Loads up the lovely dataset
df = pd.read_csv(r'C:/Users/Clyde_glhf/Desktop/Hate Speech ML App/Capstone_Project/Data_Set_Folder/Capstone Dataset.csv')

# Checking for class distribution, saved for debugging purposes
# print("Class distribution in the dataset:")
# print(df['class'].value_counts())

#Greeting to User Message
print("\n\nHello, Welcome to our Hate Speech Detection System, picking out the bad apples so you don't have to!")
print("\nOur dataset is composed of 3 classes: 0 - representing Neither, 1 - representing Offensive Language, 2 - representing Hate Speech")

# Visualization 1: Bar Chart Class Distribution
def plot_class_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    print("\nVisualization #1, Bar Chart Class Distribution: Please close the figure when satisfied to continue to the next step!")
    plt.show()
   
plot_class_distribution(df)

# Step 2: Clean and prepare the data
X = df['tweet']
y = df['class']  # Use the class column directly

# Oversample the minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)
X_resampled = X_resampled.flatten()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("\nData preprocessing complete.")

# Step 3: Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)
print("Model training complete.")

# Step 4: Test the model
y_pred = nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization 2: Confusion Matrix 
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hate Speech', 'Offensive Language', 'Neither'], yticklabels=['Hate Speech', 'Offensive Language', 'Neither'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print("\nVisualization #2, Confusion Matrix: Please close the figure when satisfied to continue to the next step!")
    plt.show()
    

plot_confusion_matrix(y_test, y_pred)

# Visualization 3: ROC Curve
def plot_roc_curve(y_test, y_pred_proba, n_classes):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    print("\nVisualization #3, ROC Curve: Please close the figure when satisfied to continue to the next step in our program!")
    plt.show()

y_pred_proba = nb_classifier.predict_proba(X_test_vec)
plot_roc_curve(y_test, y_pred_proba, n_classes=3)

# Step 5: User Interface with Random Posts, includes classification and Recommended actions
def classify_tweet(tweet):
    tweet_vec = vectorizer.transform([tweet])
    prediction = nb_classifier.predict(tweet_vec)
    if prediction == 0:
        return "Hate Speech, recommendation to remove"
    elif prediction == 1:
        return "Offensive Language, recommended for human review"
    else:
        return "Neither, no action needed"

print("\nWelcome to our Text Classification UI")
print("Type 'exit' to quit the program.")
while True:
    user_input = input("\nType 'next' to see a post with our diagnostic recommendation or 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'next':
        random_index = random.randint(0, len(df) - 1)
        random_tweet = df.iloc[random_index]['tweet']
        result = classify_tweet(random_tweet)
        print(f"Tweet: {random_tweet}")
        print(f"Classification: {result}")
    else:
        print("\nInvalid input. Please type 'next' or 'exit'.")
print("\nThank you for using our Hate Speech Detection System, have a great day!\n\n")