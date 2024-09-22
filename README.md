# Fraud-Transaction-Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv(r'C:\power bi\Fraud Transaction Detection\creditcard.csv')

# Preprocess the data
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Output the evaluation report
report_df

![image](https://github.com/user-attachments/assets/594e7b78-faa2-4424-9161-32b6cce15040)
