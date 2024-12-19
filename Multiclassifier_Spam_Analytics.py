#STTHK3013 - Pattern Recognition & Analysis
#Enhanced Multiclassifier with Highest Voting Algorithm

# Loading the dataset
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('spam.csv')

# 1. Handling missing values
print("Missing Values Before:")
print(df.isnull().sum())
df.dropna(inplace=True)  # Drop rows with missing values
print("Missing Values After:")
print(df.isnull().sum())

# Exploratory data analysis
sns.countplot(data=df, x=df["type"]).set_title("Distribution of Spam and Non-Spam Messages", fontweight="bold")
plt.show()

# Encoding labels
df['label_num'] = df['type'].apply(lambda x: 1 if x == 'spam' else 0)

# Splitting the data (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

# 2. Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, criterion='entropy'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC(probability=True, kernel='linear'),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB(),
    'MLP Neural Network': MLPClassifier(solver='adam', activation='relu', alpha=1e-05, tol = 1e-04, hidden_layer_sizes=(6,),random_state=1, max_iter = 1000),
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(train_transformed, y_train)
    y_pred = clf.predict(test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n---------------------- {name} ----------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
print('Performing GridSearchCV for MLP Classifier')
# Perform GridSearchCV for MLP Classifier
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'max_iter': [500]
}
mlp_grid = GridSearchCV(MLPClassifier(solver='adam', activation='relu', alpha=1e-05, tol = 1e-04, hidden_layer_sizes=(6,),random_state=1, max_iter = 1000), mlp_param_grid, cv=3, scoring='accuracy')
mlp_grid.fit(train_transformed, y_train)
print("Best Parameters for MLP:", mlp_grid.best_params_)

# Add the best MLP to classifiers
classifiers['MLP Neural Network'] = mlp_grid.best_estimator_

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(train_transformed, y_train)
    y_pred = clf.predict(test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n---------------------- {name} ----------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# 3. Voting Classifier
voting_clf = VotingClassifier(
    estimators=[(name, clf) for name, clf in classifiers.items()],
    voting='hard'
)
voting_clf.fit(train_transformed, y_train)
y_pred_voting = voting_clf.predict(test_transformed)

# Evaluate Voting Classifier
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print("\n==================== Voting Classifier ====================")
print(f"Accuracy: {voting_accuracy * 100:.2f}%")
print(f"Classification Report:\n{classification_report(y_test, y_pred_voting)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_voting)}")

# Final Test: Predict new messages
new_test_message = ["Congratulations! You've won $10,000. Click this link to claim your prize!"]
message_transformed = vectorizer.transform(new_test_message)
prediction = voting_clf.predict(message_transformed)
result = "Spam" if prediction[0] == 1 else "Ham"
print("\n---------------- New Message Prediction ----------------")
print(f"Message: {new_test_message[0]}")
print(f"Prediction: {result}")
