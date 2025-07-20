import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.DataFrame({
    'EmployeeID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Experience': [5, 3, 10, 7, 2, 8, 4, 6],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR'],
    'Salary': [650000, 480000, 890000, 750000, 430000, 820000, 610000, 500000]
})

le = LabelEncoder()
data['Department'] = le.fit_transform(data['Department'])

X = data[['Experience', 'Department']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
