# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Create customer data
np.random.seed(42)
n_customers = 5000

data = pd.DataFrame({
    "CustomerID": range(1, n_customers + 1),
    "Age": np.random.randint(18, 65, n_customers),
    "Income": np.random.randint(20000, 150000, n_customers),
    "EmploymentStatus": np.random.choice(["Employed", "Self-Employed", "Unemployed"], n_customers, p=[0.6, 0.3, 0.1]),
    "LoanProduct": np.random.choice(["PL", "SALPL", "RPL", "GL", "CreditCard"], n_customers),
    "LoanAmount": np.random.randint(5000, 500000, n_customers),
    "RepaymentHistory": np.random.choice(["Good", "Average", "Poor"], n_customers, p=[0.7, 0.2, 0.1]),
    "Frequency": np.random.randint(1, 20, n_customers),
    "Recency": np.random.randint(1, 12, n_customers),
    "Monetary": np.random.randint(5000, 500000, n_customers),
    "LoanTaken": np.random.choice([1, 0], n_customers, p=[0.3, 0.7])  # Target variable: 1 = Loan Taken, 0 = No Loan
})

# Preview the data
print(data.head())

# Assign RFM Scores
data['R_Score'] = pd.qcut(data['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
data['F_Score'] = pd.qcut(data['Frequency'], 4, labels=[1, 2, 3, 4]).astype(int)
data['M_Score'] = pd.qcut(data['Monetary'], 4, labels=[1, 2, 3, 4]).astype(int)

# Total RFM Score
data['RFM_Score'] = data['R_Score'] + data['F_Score'] + data['M_Score']

# Segment Customers
data['RFM_Segment'] = pd.cut(data['RFM_Score'], bins=[2, 6, 9, 12], labels=['Low', 'Medium', 'High'])

# Analyze Segments
rfm_summary = data.groupby('RFM_Segment').agg({
    'LoanTaken': ['mean', 'count']
}).reset_index()

print(rfm_summary)

from sklearn.preprocessing import LabelEncoder

# Encode categorical features
categorical_features = ['EmploymentStatus', 'LoanProduct', 'RepaymentHistory']
encoder = LabelEncoder()
for col in categorical_features:
    data[col] = encoder.fit_transform(data[col])

# Binning Age and Income
data['AgeGroup'] = pd.cut(data['Age'], bins=[18, 25, 35, 45, 55, 65], labels=[1, 2, 3, 4, 5])
data['IncomeBracket'] = pd.qcut(data['Income'], 4, labels=[1, 2, 3, 4])

# Drop unnecessary columns
X = data.drop(columns=['CustomerID', 'LoanTaken', 'RFM_Score', 'RFM_Segment'])
y = data['LoanTaken']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict probabilities
X_test['LoanProbability'] = model.predict_proba(X_test)[:, 1]

# Sort by probability
X_test['Decile'] = pd.qcut(X_test['LoanProbability'], 10, labels=list(range(10, 0, -1)))

# Decile Summary
decile_summary = X_test.groupby('Decile')['LoanProbability'].agg(['count', 'mean']).reset_index()
print(decile_summary)

# Plot Decile Analysis
sns.barplot(x='Decile', y='mean', data=decile_summary)
plt.title('Loan Probability by Decile')
plt.xlabel('Decile')
plt.ylabel('Average Loan Probability')
plt.show()

