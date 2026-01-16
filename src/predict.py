import joblib
import pandas as pd
import os

def predict_species(bill_length, bill_depth, flipper_length, body_mass, island, sex):
    """
    Toma medidas físicas y devuelve la especie predicha.
    """
    # Obtener la ruta absoluta de la carpeta donde está este script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, '..', 'penguin_model.pkl')
    encoder_path = os.path.join(BASE_DIR, '..', 'species_encoder.pkl')

    # 1. Cargar el modelo y los codificadores
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except FileNotFoundError:
        return f"Error: No se encontró el modelo en {model_path}. Revisa la ubicación."

    # 2. Preparar los datos de entrada
    # Nota: Deben tener el mismo formato que X_train (incluyendo One-Hot Encoding)
    data = {
        'bill_length_mm': [bill_length],
        'bill_depth_mm': [bill_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'island_Dream': [1 if island.lower() == 'dream' else 0],
        'island_Torgersen': [1 if island.lower() == 'torgersen' else 0],
        'sex_Male': [1 if sex.lower() == 'male' else 0]
    }
    
    input_df = pd.DataFrame(data)

    # 3. Realizar la predicción
    prediction_numeric = model.predict(input_df)
    species_name = label_encoder.inverse_transform(prediction_numeric)

    return species_name[0]

if __name__ == "__main__":
    print("--- Sistema de Clasificación de Pingüinos ---")
    # Ejemplo de prueba con datos típicos de un Gentoo
    especie = predict_species(50.0, 15.0, 220.0, 5000.0, 'Biscoe', 'Male')
    print(f"Resultado de la clasificación: {especie}")