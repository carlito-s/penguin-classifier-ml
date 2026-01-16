import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
import joblib  

from preprocessing import X_train, X_test, y_train, y_test, target_names, label_encoder 

def train_penguin_model():     
    """Entrena el modelo y lo guarda en disco."""     
    print("Iniciando entrenamiento del Random Forest...")          
    
    # 1. Instanciar y Entrenar     
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
    rf_model.fit(X_train, y_train) 
    
    # 2. Evaluación     
    predictions = rf_model.predict(X_test)
    print("\n--- Desempeño del Modelo ---")     
    print(classification_report(y_test, predictions, target_names=target_names)) 
    
    # 3. Guardar el modelo y el encoder (Serialización)
    # Usamos joblib para persistir ambos objetos
    joblib.dump(rf_model, 'penguin_model.pkl') 
    joblib.dump(label_encoder, 'species_encoder.pkl') # Ahora sí funcionará [cite: 2]
    print("\n✅ Modelo y Encoder guardados exitosamente.")          
    
    return rf_model  

def plot_feature_importance(model, features):     
    """Genera un gráfico de qué variables son más importantes."""     
    importances = model.feature_importances_ 
    plt.figure(figsize=(10, 6))   
    sns.barplot(x=importances, y=features) 
    plt.title("Importancia de las Características en la Clasificación")     
    plt.show()  

if __name__ == "__main__":     
    trained_model = train_penguin_model() 
    plot_feature_importance(trained_model, X_train.columns) 