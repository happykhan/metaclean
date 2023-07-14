import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import Pipeline
import typer

def main(dataframe: str = 'full_frame.gz'):
    # open full_frame.gz
    df1 = pd.read_csv(dataframe, compression='gzip', header=0, sep=',', quotechar='"')

    # Define a list of embedding methods to evaluate
    embedding_var= ['tfidf', 'mpnet', 'glove', 'word2vec']

    # Define a list of classifier models to use
    classifiers = [('rf', RandomForestClassifier(random_state=76)),
                    ('svm', SVC(random_state=76)), 
                    ('lr', LogisticRegression(random_state=76, max_iter=400)),
                    ('dt', DecisionTreeClassifier(random_state=76))]

    # Define a dictionary to store accuracy results for each classifier
    accuracy_lists = {
        'rf': [],
        'svm': [],
        'lr': [],
        'dt': []
    }

    # Convert source type to numeric
    le = LabelEncoder().fit(list(set(df1['EB Source Type'])))
    df1['Source type no'] =  le.transform(df1['EB Source Type'])

    # Loop through each embedding method
    for emb in embedding_var:
        # Convert string vectors to numeric vectors
        df1[emb] = df1[emb].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        # Split the data into training and testing sets using the 'train_test_split' function
        X_train, X_test, y_train, y_test = train_test_split(
            df1[emb],
            df1['Source type no'],
            test_size=0.25,
            random_state=76
        )

        # Stack the training and testing sets into 3D arrays
        X_train_stacked = np.stack(X_train)
        X_test_stacked = np.stack(X_test)

        # Loop through each classifier model
        for classifier_name, classifier in classifiers:

            # Create a pipeline that scales the data and fits the classifier
            pipe = Pipeline([('scaler', RobustScaler()), (classifier_name, classifier)])
            pipe.fit(X_train_stacked, y_train)

            # Use the pipeline to make predictions on the test data
            y_pred = pipe.predict(X_test_stacked)

            # Evaluate the accuracy of the predictions
            report = classification_report(y_test, y_pred ,output_dict=True)
            acc = report['accuracy']

            # Store the accuracy results for each classifier
            accuracy_lists[classifier_name].append(acc)

    # Add a new key 'embeddings' to the dictionary 'accuracy_lists' and assign the list 'embedding_var' to it
    accuracy_lists['embeddings'] = embedding_var

    # Create a list of tuples using the values from the dictionaries
    df_zip = list(zip(accuracy_lists['embeddings'], accuracy_lists['lr'], accuracy_lists['svm'], accuracy_lists['rf'], accuracy_lists['dt']))

    # Create a DataFrame 'df_accuracy' from the list 'df_zip' and specify the column names
    df_accuracy = pd.DataFrame(df_zip, columns = ['Embedding','Logistic_Regression','Support_Vector_Machine', 'Random_Forest','Decision_Tree'])

    # Print the DataFrame 'df_accuracy'
    print(df_accuracy)

    # write df_accuracy to csv
    df_accuracy.to_csv('df_accuracy.csv', index=False)


if __name__ == "__main__":
    typer.run(main)

