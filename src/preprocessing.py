import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Carga
df = sns.load_dataset('penguins')

# 2. Manejo de valores nulos (Imputación)
# Usamos la mediana para variables numéricas para evitar sesgos por outliers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separamos columnas por tipo
num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_cols = ['island', 'sex']

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 3. Codificación de variables categóricas (Features)
# Usamos get_dummies para crear variables temporales (One-Hot Encoding)
df_final = pd.get_dummies(df, columns=['island', 'sex'], drop_first=True)

# 4. Codificación de la variable objetivo (Target)
# Convertimos 'Adelie', 'Chinstrap', 'Gentoo' en 0, 1, 2
label_encoder = LabelEncoder()
df_final['species'] = label_encoder.fit_transform(df_final['species'])

# 5. División del Dataset: Entrenamiento y Prueba (Hold-out Method)
X = df_final.drop('species', axis=1)
y = df_final['species']

# Reservamos el 20% de los datos para la "prueba final" (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Importante: para que model.py sepa los nombres de las especies
target_names = label_encoder.classes_

print(f"Set de entrenamiento: {X_train.shape[0]} pingüinos")
print(f"Set de prueba: {X_test.shape[0]} pingüinos")