# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings


def multiclass_specificity(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    num_classes = conf_matrix.shape[0]
    specificity_values = []
    for i in range(num_classes):
        # Calcula la especificidad para la clase i
        tn = np.sum(conf_matrix[i, j] for j in range(num_classes) if j != i)
        fp = np.sum(conf_matrix[j, i] for j in range(num_classes) if j != i)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Evita divisiones por cero
        specificity_values.append(specificity)
    # Calcula la especificidad promediada
    mean_specificity = np.mean(specificity_values)
    return mean_specificity

def modelDataSet(dataSetName):
    data = pd.read_csv(dataSetName)
    X=data.iloc[:, 1:-1]
    y=data.iloc[:, -1]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalizar los datos (solo para algoritmos sensibles a la escala como KNN y SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Inicializar los modelos
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machines': SVC(),
        'Naive Bayes': GaussianNB(),
    }
    # Entrenar y evaluar los modelos
    # Dentro de la función modelDataSet justo antes de entrar al bucle for
    warnings.filterwarnings("ignore")
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled if 'K-Nearest Neighbors' in name or 'Support Vector Machines' in name else X_train, y_train)
        y_pred = model.predict(X_test_scaled if 'K-Nearest Neighbors' in name or 'Support Vector Machines' else X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) 
        recall = recall_score(y_test, y_pred, average='weighted',zero_division=1)
        f1 = f1_score(y_test, y_pred,  average='weighted')
        specificity = multiclass_specificity(y_test, y_pred)  # Calcula la especificidad promediada
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity
        }
        
    print(f'================{dataSetName}================')
    for name, result in results.items():
        print(f"Results for {name}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")
        print("\n")




modelDataSet("zoo2.csv")
modelDataSet("zoo3.csv")
    

