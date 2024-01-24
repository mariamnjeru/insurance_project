import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('insurance 2.csv')

# Display the initial few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Encode 'sex' using label encoding
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])

# Encode 'smoker' using label encoding
df['smoker'] = label_encoder.fit_transform(df['smoker'])

# One-hot encode 'region'
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Display the processed dataset
print("\nProcessed Dataset:")
print(df.head())


from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Instantiate the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


import streamlit as st

# Streamlit app
st.title("Health Insurance Cost Predictor")

# Input fields for user to enter information
age = st.slider("Age:", min_value=18, max_value=64, value=30)
sex = st.radio("Gender:", options=['Female', 'Male'])
bmi = st.slider("BMI:", min_value=15.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children:", min_value=0, max_value=5, value=0)
smoker = st.radio("Smoker:", options=['No', 'Yes'])
region = st.radio("Region:", options=['Northwest', 'Southeast', 'Southwest'])

# Encode categorical inputs
sex_encoded = 0 if sex == 'Female' else 1
smoker_encoded = 1 if smoker == 'Yes' else 0

# Map region to one-hot encoding
region_encoded = {
    'Northwest': [1, 0, 0],
    'Southeast': [0, 1, 0],
    'Southwest': [0, 0, 1]
}
region_encoded = region_encoded.get(region, [0, 0, 0])

# Make a prediction
user_input = [[age, sex_encoded, bmi, children, smoker_encoded] + region_encoded]
prediction = model.predict(user_input)[0]

# Display the predicted cost
st.subheader("Predicted Health Insurance Cost:")
st.write(f"${prediction:.2f}")

