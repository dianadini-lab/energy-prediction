import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("Mulai memproses data...")

# 1. Membaca file CSV datasheet
df = pd.read_csv("Energy_consumption.csv")

# 2. Mengambil fitur jam (Hour) dari kolom Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# 3. Menentukan kolom fitur (X) dan target (y)
features = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'RenewableEnergy', 'HVACUsage', 'LightingUsage', 'Holiday', 'Hour'
]
target = 'EnergyConsumption'

X = df[features]
y = df[target]

# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Membuat Preprocessor & Pipeline Linear Regression
numerical_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy', 'Hour']
categorical_features = ['HVACUsage', 'LightingUsage', 'Holiday']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Melatih model dan menyimpan hasilnya menjadi file .pkl
lr_pipeline.fit(X_train, y_train)
joblib.dump(lr_pipeline, 'model_linear_regression.pkl')

print("SUKSES: Model berhasil dilatih dan file 'model_linear_regression.pkl' sudah dibuat di folder ini!")