import pandas as pd

df = pd.read_csv(r"C:\Users\k18at\OneDrive\Documents\Task 3\housing_prices.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_features = ['Lot Area', 'Overall Qual', 'Year Built']
cat_features = ['Neighborhood', 'House Style']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X = df[num_features + cat_features]
y = df['SalePrice']
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
import joblib
joblib.dump(model, 'house_price_model.pkl')