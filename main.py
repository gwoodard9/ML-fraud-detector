import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

model = LogisticRegression(max_iter=1000, solver='liblinear')
warnings.filterwarnings("ignore", category=FutureWarning)

def load_file(file_path):
    if os.path.exists(file_path):
        try:
            _, file_extension = os.path.splitext(file_path)
            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows = 200)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path, nrows = 200)
            else:
                print('Unsupported file type. Please upload a .csv or .xlsx file.')
                return None
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print('File does not exist. Please check file path and try again.')
        return None
    
def preprocess_data(df):
    df = df.fillna(0)
    
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])
    
    df['isFraud'] = df['isFraud'].astype(int)
    
    return df

def load_data():
    df = pd.read_csv('/Users/gagewoodard/Downloads/cardsim.csv')
    return df

def check_missing_data(df):
    print(f"\nMissing data per column:\n{df.isnull().sum()}\n")

def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean())

def detect_transaction_type(df):
    possible_columns = ['type', 'transaction_type', 'category']
    for col in possible_columns:
        if col in df.columns:
            return df[col]
    return None

def encode_categorical_columns(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            print(f"Encoding column: {column}")
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
    return df

def split_features_target(df):
    X = df.drop(columns=['isFraud'])
    Y = df['isFraud']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test

def handle_class_imbalance(X_train, y_train):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def main():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_scaled, y_train)

    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()