pip install scikit-learn pandas numpy python-taiga

pip install python-taiga

import pandas as pd
import numpy as np

# Generate dummy task data
np.random.seed(42)

num_tasks = 100
start_dates = pd.date_range(start="2025-01-01", periods=num_tasks, freq="D")
priorities = np.random.choice(["Low", "Medium", "High"], size=num_tasks)
complexities = np.random.randint(1, 10, size=num_tasks)  # Complexity from 1 to 10
durations = complexities * np.random.uniform(0.8, 1.5, size=num_tasks) + \
            (priorities == "High") * np.random.uniform(2, 5, size=num_tasks)

# Create DataFrame
df = pd.DataFrame({
    "task_id": range(1, num_tasks + 1),
    "start_date": start_dates,
    "priority": priorities,
    "complexity": complexities,
    "duration": durations.round(2)  # Duration in days
})

print(df.head())


#######################


from sklearn.model_selection import train_test_split

# Encode priority as numerical values
df["priority_encoded"] = df["priority"].map({"Low": 1, "Medium": 2, "High": 3})

# Convert start_date to days since project start
project_start = df["start_date"].min()
df["days_since_start"] = (df["start_date"] - project_start).dt.days

# Define features (X) and target (y)
X = df[["days_since_start", "priority_encoded", "complexity"]]
y = df["duration"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


########################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


#############################

# Simulate new tasks
new_tasks = pd.DataFrame({
    "days_since_start": [110, 120],
    "priority_encoded": [3, 2],  # High and Medium priority
    "complexity": [8, 5]
})

# Predict durations for new tasks
predicted_durations = model.predict(new_tasks)
new_tasks["predicted_duration"] = predicted_durations.round(2)

print(new_tasks)

#################################


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.xlabel("Actual Duration")
plt.ylabel("Predicted Duration")
plt.title("Actual vs Predicted Task Durations")
plt.show()

####################################


import taiga
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize Taiga API
api = taiga.TaigaAPI(host="https://api.taiga.io")
api.auth(username="your_username", password="your_password")

# Fetch tasks
project_id = 12345
tasks = api.tasks.list(project=project_id)

# Prepare data
data = {
    "task_id": [],
    "duration": [],
    "start_date": [],
    "end_date": [],
}

for task in tasks:
    data["task_id"].append(task.id)
    data["duration"].append((task.finished_date - task.start_date).days if task.finished_date else None)
    data["start_date"].append(task.start_date)
    data["end_date"].append(task.finished_date)

df = pd.DataFrame(data)
df = df.dropna()

# Convert start_date to numerical feature
project_start = df["start_date"].min()
df["days_since_start"] = (df["start_date"] - project_start).dt.days

# Split data
X = df[["days_since_start"]]
y = df["duration"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse}")

# Predict new task duration
new_task_start_date = pd.to_datetime("2025-04-15")
new_task_days_since_start = (new_task_start_date - project_start).days
new_task_duration = model.predict([[new_task_days_since_start]])
print(f"Predicted duration for new task: {new_task_duration[0]} days")

# Update Taiga task due date
task_id = 67890
new_due_date = new_task_start_date + pd.Timedelta(days=new_task_duration[0])
task = api.tasks.get(task_id)
task.due_date = new_due_date
api.tasks.update(task_id, due_date=task.due_date)
