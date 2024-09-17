import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, 
                             confusion_matrix, recall_score, 
                             f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
email_data = pd.read_csv('emails.csv')
print(email_data.head())
email_data.info()

# Check for missing values and data dimensions
print(email_data.isnull().sum())
print(email_data.shape)

# Remove any rows with missing data
clean_data = email_data.dropna()
print(clean_data)

# Prepare features and labels
features = clean_data.drop(columns=['Email No.', 'Prediction'])
labels = clean_data['Prediction']
print('Features shape:', features.shape)
print('Labels shape:', labels.shape)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.4, random_state=44)

# Train the Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)

# Make predictions
Y_pred = nb_classifier.predict(X_test)

# Confusion matrix visualization
conf_matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model
print('Classification Report:')
print(classification_report(Y_test, Y_pred))
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
