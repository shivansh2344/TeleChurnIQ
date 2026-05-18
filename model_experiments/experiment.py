import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style for professional reports
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Configure logging
LOG_DIR = 'model_experiments/logs'
RESULTS_DIR = 'model_experiments/results'
MODELS_DIR = 'model_experiments/models'
EDA_DIR = 'model_experiments/eda'

for d in [LOG_DIR, RESULTS_DIR, MODELS_DIR, EDA_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """Loads dataset from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df

def preprocess_data(df, target_col='Churn'):
    """Performs preprocessing: handles missing values, encoding, and scaling."""
    logging.info("Starting preprocessing...")
    
    # 1. Handle Missing Values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # 2. Encode Categorical Variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    if df[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str))
        label_encoders[target_col] = le_target
    
    # 3. Data Augmentation (Oversampling)
    churn_count = df[df[target_col] == 1].shape[0]
    no_churn_count = df[df[target_col] == 0].shape[0]
    
    if churn_count < no_churn_count:
        df_churn = df[df[target_col] == 1]
        df_churn_over = df_churn.sample(no_churn_count, replace=True, random_state=42)
        df_balanced = pd.concat([df[df[target_col] == 0], df_churn_over], axis=0)
    else:
        df_no_churn = df[df[target_col] == 0]
        df_no_churn_over = df_no_churn.sample(churn_count, replace=True, random_state=42)
        df_balanced = pd.concat([df[df[target_col] == 1], df_no_churn_over], axis=0)
    
    logging.info(f"Data balanced. New shape: {df_balanced.shape}")
    
    X = df_balanced.drop(columns=[target_col])
    y = df_balanced[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_columns = X.columns.tolist()
    
    logging.info("Preprocessing complete.")
    return X_scaled, y, X_columns, scaler, label_encoders

def perform_eda(df, target_col='Churn'):
    """Generates exploratory data analysis graphs."""
    logging.info("Generating EDA visual reports...")
    
    # 1. Churn Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_col, palette='viridis')
    plt.title('Distribution of Customer Churn', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status (0: Stay, 1: Churn)')
    plt.ylabel('Customer Count')
    plt.savefig(os.path.join(EDA_DIR, 'churn_distribution.png'), dpi=300)
    plt.close()

    # 2. Correlation Matrix (Numeric Features)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(EDA_DIR, 'correlation_matrix.png'), dpi=300)
        plt.close()

    # 3. Key Numerical Distributions by Churn
    key_numeric = ['Age', 'Tenure', 'MonthlyCharge', 'TenureinMonths']
    for col in key_numeric:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df, x=col, hue=target_col, fill=True, palette='magma')
            plt.title(f'Distribution of {col} by Churn Status', fontsize=14, fontweight='bold')
            plt.savefig(os.path.join(EDA_DIR, f'dist_{col.lower()}.png'), dpi=300)
            plt.close()

    # 4. Categorical Impact (Contract/Gender)
    key_cat = ['Contract', 'Gender', 'InternetService']
    for col in key_cat:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=col, hue=target_col, palette='Set2')
            plt.title(f'Impact of {col} on Churn', fontsize=14, fontweight='bold')
            plt.legend(title='Churn', labels=['Stay', 'Churn'])
            plt.savefig(os.path.join(EDA_DIR, f'impact_{col.lower()}.png'), dpi=300)
            plt.close()

def save_visual_metrics(results_df):
    """Generates professional bar charts for model comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # 1. Overall Metrics Bar Chart
    df_melted = results_df.melt(id_vars='model', value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=df_melted, x='model', y='Score', hue='Metric', palette='viridis')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.xlabel('Algorithm')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add data labels
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=9)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'overall_comparison_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. F1-Score Specific Ranking (Clean PNG Table replacement)
    plt.figure(figsize=(10, 6))
    ranking = results_df.sort_values(by='f1_score', ascending=False)
    sns.barplot(data=ranking, x='f1_score', y='model', palette='magma')
    plt.title('Algorithm Ranking (by F1-Score)', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.1)
    plt.xlabel('F1-Score')
    plt.ylabel('Model')
    plt.savefig(os.path.join(RESULTS_DIR, 'f1_ranking_report.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """Trains and evaluates a single model and saves its confusion matrix."""
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Save Professional Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, pad=15)
    plt.ylabel('Actual Customer Status (0: Stay, 1: Churn)', fontsize=12)
    plt.xlabel('Predicted Customer Status', fontsize=12)
    plt.xticks([0.5, 1.5], ['Stay', 'Churn'])
    plt.yticks([0.5, 1.5], ['Stay', 'Churn'])
    
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{model_name.lower().replace(" ", "_")}.png'), dpi=300)
    plt.close()
    
    logging.info(f"{model_name} metrics: {metrics}")
    return metrics, model

def predict_single(input_dict, scaler, label_encoders, feature_names, best_model):
    """Single customer inference with preprocessing."""
    df_input = pd.DataFrame([input_dict])
    for col, le in label_encoders.items():
        if col in df_input.columns and col != 'Churn':
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except ValueError:
                df_input[col] = 0 
    df_input = df_input[feature_names]
    X_single = scaler.transform(df_input)
    pred = best_model.predict(X_single)[0]
    prob = best_model.predict_proba(X_single)[0][1]
    return int(pred), float(prob)

def run_experiment(csv_path):
    """Full experimentation pipeline."""
    df = load_data(csv_path)
    
    # Run EDA before preprocessing
    perform_eda(df)
    
    X, y, feature_names, scaler, encoders = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        metrics, trained_model = train_and_evaluate(X_train, X_test, y_train, y_test, name, model)
        results.append(metrics)
        trained_models[name] = trained_model
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
    
    # Generate visual reports
    save_visual_metrics(results_df)
    
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    best_model_name = 'XGBoost' if 'XGBoost' in trained_models else results_df.sort_values(by='f1_score', ascending=False).iloc[0]['model']
    best_model = trained_models[best_model_name]
    
    print(f"\n>>> Selected Best Model: {best_model_name}")
    
    final_bundle = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': feature_names,
        'model_name': best_model_name,
        'timestamp': datetime.now().isoformat()
    }
    joblib.dump(final_bundle, os.path.join(MODELS_DIR, 'churn_model.pkl'))
    logging.info(f"Final model saved to {MODELS_DIR}/churn_model.pkl")
    
    return final_bundle

if __name__ == "__main__":
    DATA_PATH = 'frontend/public/data/telco.csv'
    
    if not os.path.exists(DATA_PATH):
        print(f"Warning: {DATA_PATH} not found. Creating dummy data...")
        dummy_data = pd.DataFrame({
            'Age': np.random.randint(18, 80, 1000),
            'Tenure': np.random.randint(0, 72, 1000),
            'MonthlyCharge': np.random.uniform(20, 120, 1000),
            'Contract': np.random.randint(0, 3, 1000),
            'Gender': np.random.randint(0, 2, 1000),
            'Churn': np.random.choice([0, 1], 1000)
        })
        dummy_data.to_csv('model_experiments/dummy_telco.csv', index=False)
        DATA_PATH = 'model_experiments/dummy_telco.csv'

    print(f"Starting experiment with dataset: {DATA_PATH}")
    best_bundle = run_experiment(DATA_PATH)
    
    # Test customer for prediction verification
    test_customer = {
        'Age': 45, 'Tenure': 12, 'MonthlyCharge': 85.5, 'Contract': 0, 'Gender': 0
    }
    for feat in best_bundle['features']:
        if feat not in test_customer: test_customer[feat] = 0
            
    prediction, probability = predict_single(
        test_customer, best_bundle['scaler'], best_bundle['encoders'], 
        best_bundle['features'], best_bundle['model']
    )
    
    print(f"\n--- Single Prediction Test ---")
    print(f"Prediction: {'CHURN' if prediction == 1 else 'STAY'}")
    print(f"Probability: {probability:.4f}")
    print(f"\nVisual reports saved in: {RESULTS_DIR}")
