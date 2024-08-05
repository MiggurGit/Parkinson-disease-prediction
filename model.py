import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib

def train_model():
    # Load dataset
    data = pd.read_csv('parkinsons.csv')
    
    # Features and target
    X = data.drop(columns=['status'])
    y = data['status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with a scaler and SVM
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(pipeline, 'parkinsons_model.pkl')

if __name__ == "__main__":
    train_model()
