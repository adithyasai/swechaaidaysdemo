pip install fairlearn

pip install scikit-learn pandas python-redmine matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Dummy Data
np.random.seed(0)  # For reproducibility

# Team Member Information
team_members = ['John', 'Emily', 'Michael', 'Sarah', 'William', 'Olivia', 'James', 'Ava', 'George', 'Isabella']
genders = ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
roles = ['Developer', 'Manager', 'Developer', 'Manager', 'Developer', 'Manager', 'Developer', 'Manager', 'Developer', 'Manager']

# Task Assignment Data
task_assignments = np.random.randint(0, 2, size=100)  # 100 tasks, randomly assigned

# Task Types
task_types = np.random.choice(['Development', 'Management', 'Testing'], size=100)

# Create DataFrame
data = {
    'team_member': np.random.choice(team_members, size=100),
    'gender': np.random.choice(genders, size=100),
    'role': np.random.choice(roles, size=100),
    'task_type': task_types,
    'task_assigned': task_assignments
}
df = pd.DataFrame(data)

# Convert categorical variables into numerical values
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
le = LabelEncoder()
df['role_encoded'] = le.fit_transform(df['role'])
df['task_type_encoded'] = le.fit_transform(df['task_type'])

# Split data into training and testing sets
X = df[['role_encoded', 'task_type_encoded']]  # Use encoded features
y = df['task_assigned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Bias Detection using Fairlearn
sensitive_features = df['gender'].iloc[X_test.index]
bias_score = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)

print(f"Demographic Parity Difference: {bias_score}")

# Visualize Task Distribution by Role and Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='role', hue='gender')
plt.title('Task Distribution by Role and Gender')
plt.show()

# Visualize Task Distribution by Task Type and Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='task_type', hue='gender')
plt.title('Task Distribution by Task Type and Gender')
plt.show()

# Visualize Bias Metric
plt.figure(figsize=(8, 6))
plt.bar(['Male', 'Female'], [df[df['gender'] == 0]['task_assigned'].mean(), df[df['gender'] == 1]['task_assigned'].mean()])
plt.title('Average Task Assignment by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Tasks Assigned')
plt.show()