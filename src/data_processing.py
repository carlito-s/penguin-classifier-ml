import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def get_processed_data():
    """Carga, limpia y divide los datos de los pingüinos."""
    df = sns.load_dataset('penguins')
    
    # Imputación simple
    num_cols = df.select_dtypes(include=['float64']).columns
    cat_cols = ['island', 'sex']
    
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    
    # Codificación de la meta (Target)
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    
    # Preparación de Features (One-Hot Encoding)
    # Importante: Mantener consistencia con el orden de columnas para la app
    df_final = pd.get_dummies(df, columns=['island', 'sex'], drop_first=True)
    
    X = df_final.drop('species', axis=1)
    y = df_final['species']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), le, X.columns