import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load data
df = pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Randomforestclassifierexample/500hits.csv", encoding='latin-1')
print(df.head())
# Check for missing values
print(df.isnull().sum())
# Drop columns
df = df.drop(columns=['PLAYER', 'CS'])
# Define features and target
X = df.iloc[:, 0:13]
Y = df.iloc[:, 13]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Initialize and train the model
rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=42)
rf.fit(X_train, y_train)
# Predictions
y_pred = rf.predict(X_test)
# Evaluate
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
