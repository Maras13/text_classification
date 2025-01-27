
import numpy as np
import pandas as pd




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

import joblib


tfidf_df = pd.read_csv("tfidf_df.csv")


y = tfidf_df['index']
X = tfidf_df.drop(columns="index")

# random_state seed value for the random number generator
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.20,random_state=141)


models = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "param_grid": {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
    },
    "Multinomial Naive Bayes": {
        "model": MultinomialNB(),
        "param_grid": {'alpha': [0.1, 1, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=141),
        "param_grid": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    }
}

for model_name, model_info in models.items():
    print(f"\nPerforming GridSearch for {model_name}...")

    grid_search = GridSearchCV(model_info["model"], model_info["param_grid"], cv=5, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)

    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy for {model_name}: {grid_search.best_score_}")

    # Evaluate the model on the test set
    ypred = grid_search.best_estimator_.predict(Xtest)
    print(f"Test Accuracy for {model_name}: {accuracy_score(ytest, ypred)}")
    print(f"Classification Report for {model_name}:\n", classification_report(ytest, ypred))

    # Save the best model
    best_model_filename = f"{model_name.replace(' ', '_').lower()}_best_model.pkl"
    joblib.dump(grid_search.best_estimator_, best_model_filename)
    print(f"Best {model_name} model saved as: {best_model_filename}")
