import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import joblib


def augment_data(data, columns_to_augment, scale_factor=0.001, random_seed=42):
    np.random.seed(random_seed)
    augmented_data = data.copy()
    for column in columns_to_augment:
        original_column = data[column].values.reshape(-1, 1)
        noise = np.random.normal(0, scale_factor * np.std(original_column), original_column.shape)
        augmented_column = original_column + noise
        augmented_data[column] = augmented_column.reshape(-1)
    return augmented_data


df = pd.read_csv("C:\\Users\\Narayanakshay\\Downloads\\forestfires.csv")
pd.set_option('display.max_columns', None)
print(df.head(20))
print(df.describe())
df = df.drop(['X','Y','month','day','FFMC','DMC','rain','ISI'],axis=1)
columns_to_augment = ['temp', 'RH', 'wind', 'DC', 'area']
augmented_data = pd.DataFrame()
for i in range(10):
    
    augmented_dataset1 = augment_data(df, columns_to_augment)
    augmented_data = pd.concat([augmented_dataset1,augmented_data], ignore_index=True)
df = pd.concat([df,augmented_data], ignore_index=True)
print("after augment")
print(df.describe())
X = df[['temp', 'RH', 'wind', 'DC']]
y = df['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
joblib.dump(decision_tree_model, 'escendo_model.joblib')
