import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("The specified file does not exist.")
    
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx only.")

def analyze_columns(df):
    print("----- Data Insights for Each Column -----\n")
    for column in df.columns:
        print(f"Column: {column}")
        print(f" - Data Type: {df[column].dtype}")
        print(f" - Unique Values: {df[column].nunique()}")
        print(f" - Missing Values: {df[column].isnull().sum()}")
        
        if df[column].dtype in ['int64', 'float64']:
            print(df[column].describe())
        else:
            print(f" - Top 5 Unique Values: {df[column].value_counts().head(5)}")
        print("-" * 50)
    print("----- End of Data Insights -----\n")

def preprocess_data(df):
    print("Preprocessing data...")
    
    df.fillna(0, inplace=True)
    
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    return df

def analyze_columns_and_check_target(df):
    print("----- Data Insights for Each Column -----\n")
    potential_target_columns = []
    
    for column in df.columns:
        print(f"Column: {column}")
        print(f" - Data Type: {df[column].dtype}")
        print(f" - Unique Values: {df[column].nunique()}")
        print(f" - Missing Values: {df[column].isnull().sum()}")
        
        if df[column].dtype in ['int64', 'float64']:
            print(df[column].describe())
        else:
            print(f" - Top 5 Unique Values: {df[column].value_counts().head(5)}")
        
        if df[column].nunique() == 2:
            potential_target_columns.append(column)

        print("-" * 50)
    
    print("----- End of Data Insights -----\n")
    
    if not potential_target_columns:
        raise ValueError("No binary target column found in the dataset.")
    
    print(f"Potential Target Columns: {potential_target_columns}")
    return potential_target_columns

def main():

    file_path = input("Enter the file path (CSV or XLSX): ").strip()

    try:
        df = load_dataset(file_path)
        
        potential_target_columns = analyze_columns_and_check_target(df)
        
        target_column = potential_target_columns[0]
        
        print(f"Using column '{target_column}' as the target column.")
        
        df = preprocess_data(df)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if y.empty:
            raise ValueError(f"The target column '{target_column}' is empty.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        model = LogisticRegression(max_iter=500)
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        valid_transactions = sum(y_pred == 0)
        fraud_transactions = sum(y_pred == 1)  
        
        print(f"\033[92m{valid_transactions} transactions were valid\033[0m")
        print(f"\033[91m{fraud_transactions} transactions were fraud\033[0m")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
