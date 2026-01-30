from sklearn.ensemble import RandomForestClassifier
import joblib
from data_processing import get_processed_data

def train():
    (X_train, X_test, y_train, y_test), le, feature_names = get_processed_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Guardar en la raíz
    joblib.dump(model, 'penguin_model.pkl')
    joblib.dump(le, 'species_encoder.pkl')
    print("✅ Entrenamiento completado y archivos guardados.")

if __name__ == "__main__":
    train()