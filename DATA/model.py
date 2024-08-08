import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import numpy as np
import pickle

# Load your dataset
df = pd.read_csv('casab.csv')

# Extract relevant features and target variable
X = df.drop(columns=['Price', 'Price_m2', 'Other_tags'])
y = df['Price']

# Define categorical and numerical features
categorical_features = ['Type', 'Localisation', 'Current_state', 'Age']
numerical_features = ['Area', 'Rooms', 'Bedrooms', 'Bathrooms', 'Floor']

# Create a ColumnTransformer to handle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Transform the training and test data separately
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_transformed, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_transformed)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2:.4f}')

# Save the trained model and preprocessor for future use
with open('casablanca_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('casablanca_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Define new input with the same feature structure as during training
new_input = pd.DataFrame({
    'Type': ['Villas'],
    'Localisation': ['Californie'],
    'Area': [400.0],
    'Rooms': [8.0],
    'Bedrooms': [5.0],
    'Bathrooms': [4.0],
    'Floor': [0.0],
    'Current_state': ['Bon état'],
    'Age': ['10-20 ans']
})

# Preprocess the new input
new_input_transformed = preprocessor.transform(new_input)
# Predict the output
predicted_output = model.predict(new_input_transformed)
print("Predicted output for the new input:", predicted_output[0])
print("Predicted output for the new input:",predicted_output//new_input['Area'] )
