
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

#Loading the dataset
import pandas as pd

df = pd.read_csv('spam.csv')
print(df.head())
print(df['type'].value_counts())

# 1. Handling missing values
print("Missing Values Before:")
print(df.isnull().sum())
df.dropna(inplace=True)  # Drop rows with missing values
print("Missing Values After:")
print(df.isnull().sum())

# assigning labels to the dataset, 0 for ham and 1 for spam
df['label_num'] = df['type'].apply(lambda x: 1 if x == 'spam' else 0)
print(df['label_num'])

# visualizing data distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data = df, x= df["type"]).set_title("Amount of spam and no-spam messages", fontweight = "bold")
plt.show()

# Splitting the data (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

print("train set:", X_train.shape)  # rows in train set
print("test set:", X_test.shape)  # rows in test set

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
train_transformed = vectorizer.fit_transform(X_train)
test_transformed = vectorizer.transform(X_test)

print("\nOriginal Test Set:\n",X_test)
print("\nTransformed Test Set:\n", test_transformed)

# class - classifiers
#K-Nearest Neighbour
knn = KNeighborsClassifier(n_neighbors=3)
#Random Forest
rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy',random_state =7)
#Support Vector Machine
svc = SVC(probability=True)
#Logistic Regression
lr = LogisticRegression()
#Decision Tree
dt = DecisionTreeClassifier()
#Naive Bayesian
nb = MultinomialNB()
#MLP Neural Nets
mlp = MLPClassifier(solver='adam', activation='relu', alpha=1e-05, tol = 1e-04, hidden_layer_sizes=(6,),random_state=1, max_iter = 1000)

#library for evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# making predictions on the testing set
for clf in (rfc, knn, svc,lr, dt, nb, mlp):
	clf.fit(train_transformed, y_train) #build the model
	Y_pred = clf.predict(test_transformed) #predict the test case
	print("\n----------------------Result: %s ----------------------" %(clf.__class__.__name__))
	print("\nAccuracy score of %s is %.2f%% " % (clf.__class__.__name__, 100*accuracy_score(y_test, Y_pred)))
	print("Metric classification report:", clf.__class__.__name__, "-->\n", classification_report(y_test, Y_pred))
	print("Confusion Matrix:", clf.__class__.__name__,"-->\n",confusion_matrix(y_test,Y_pred))

print('Performing GridSearchCV for MLP Classifier')
# Perform GridSearchCV for MLP Classifier
mlp_param_grid = {
    'hidden_layer_sizes': [(6,), (50,), (100,), (100, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-05],
    'tol': [1e-04],
    'max_iter': [1000]
}

mlp_grid = GridSearchCV(MLPClassifier(random_state=1), mlp_param_grid, cv=3, scoring='accuracy')
mlp_grid.fit(train_transformed, y_train)
print("Best Parameters for MLP:", mlp_grid.best_params_)

# MLP With GridSearchCV
mlpGSVC = mlp_grid.best_estimator_
#Prediction on testing set
Y_pred_gsvc = mlpGSVC.predict(test_transformed)
print(f"\n----------------------Result: MLP Neural Network (GridSearch) ----------------------")
print(f"\nAccuracy score of MLP Neural Network (GridSearch) is {100 * accuracy_score(y_test, Y_pred_gsvc):.2f}%")
print(f"Metric classification report for MLP Neural Network (GridSearch) -->\n{classification_report(y_test, Y_pred_gsvc)}")
print(f"Confusion Matrix for MLP Neural Network (GridSearch) -->\n{confusion_matrix(y_test, Y_pred_gsvc)}")

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine all models into a dictionary
all_models = {
    "Random Forest": rfc,
    "K-Nearest Neighbors": knn,
    "Support Vector Machine": svc,
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Naive Bayes": nb,
    "MLP Neural Network (Original)": mlp,
    "MLP Neural Network (GridSearch)": mlpGSVC,
}

#Voting Classifier
voting_clf = VotingClassifier(estimators=[(name, model) for name, model in all_models.items()], voting='hard')

# Fit the voting classifier on the training data
voting_clf.fit(train_transformed, y_train)

# Combine update all models into a dictionary
all_models = {
    "Random Forest": rfc,
    "K-Nearest Neighbors": knn,
    "Support Vector Machine": svc,
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Naive Bayes": nb,
    "MLP Neural Network (Original)": mlp,
    "MLP Neural Network (GridSearch)": mlpGSVC,
    "Voting Classifier": voting_clf,
    "Vectorizer": vectorizer
}

#Save the models
dump(all_models, "models_Spam.joblib")
print("Models saved successfully")

#new_messages = ["Congragulations! You have won a $10,000. Go to https://bit.ly/23343 to claim now.",
#                "Get $10 Amazon Gift Voucher on Completing the Demo:- va.pcb3.in/ click this link to claim now",
#                "You have won a $500. Please register your account today itself to claim now https://imp.com",
#                "Please dont respond to missed calls from unknown international numbers Call/ SMS on winning prize. lottery as this may be fraudulent call."]

new_test_message = ["Congragulations! You have won a $10,000. Go to https://bit.ly/23343 to claim now."]
message_transformed = vectorizer.transform(new_test_message)

case_ham = 0
case_spam = 0
total_classifier =0

print("\n ++++++++++++++++++ Predicting a new case ++++++++++++++++++++++")
print("\nNew classification for:\n", new_test_message)
for clf in (rfc, knn, svc,lr, dt, nb, mlp, mlpGSVC):
    total_classifier +=1
    Y_pred = clf.predict(message_transformed)#.reshape(1,-1))
#compute majority voting
    if Y_pred == 0:
        case_ham +=1
    else:
        case_spam +=1
    print("Classification Result:", clf.__class__.__name__,"-->",Y_pred)

#count percentage votes
percent_spam = 0
percent_ham = 0
percent_ham = (case_ham/total_classifier)*100
percent_spam = (case_spam/total_classifier)*100

#finalized based on majority voting, we will decide either a person will/not click the Ad
print("\n--------------------- Final Result ---------------------------------")
if case_ham > case_spam:
    print("Result: --> Ham e-mail. Keep or it is safe to read (based on %.2f%% vote)" % percent_ham)
else:
    print("Result: --> Spam e-mail. Delete it. It may contain harmful links (based %.2f%% vote)" % percent_spam)