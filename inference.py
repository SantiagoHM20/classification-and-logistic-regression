
import json
import pickle
import numpy as np
import os
import sys

def sigmoid(z):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip para evitar overflow

def model_fn(model_dir):
    """
    Carga el modelo desde el directorio.
    SageMaker llama a esta función cuando inicia el endpoint.
    """
    print(f"[INFO] Loading model from {model_dir}")
    
    try:
        # Cargar parámetros del modelo
        with open(os.path.join(model_dir, 'model_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        # Cargar normalización
        with open(os.path.join(model_dir, 'normalization.pkl'), 'rb') as f:
            norm_stats = pickle.load(f)
        
        # Cargar metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Features: {metadata['feature_names']}")
        print(f"[INFO] Lambda: {params['lambda']}")
        
        return {
            'params': params,
            'norm_stats': norm_stats,
            'metadata': metadata
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise


def input_fn(request_body, content_type='application/json'):
    """
    Procesa el input del request.
    Acepta formato diccionario o array.
    """
    print(f"[INFO] Content-Type: {content_type}")
    print(f"[INFO] Request body: {request_body[:200]}...") 
    
    if content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            print(f"[INFO] Parsed input: {input_data}")
            return input_data
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON: {str(e)}")
            raise ValueError(f"Invalid JSON: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Hace la predicción.
    Soporta formato diccionario y array.
    """
    print(f"[INFO] Starting prediction")
    
    try:
        params = model['params']
        norm_stats = model['norm_stats']
        metadata = model['metadata']
        
        w = params['w']
        b = params['b']
        feature_names = metadata['feature_names']
        feature_mean = norm_stats['feature_mean']
        feature_std = norm_stats['feature_std']
        
        # Formato 1: Array bajo "features"
        if "features" in input_data and isinstance(input_data["features"], list):
            print(f"[INFO] Format: Array")
            feature_array = input_data["features"]
            
            if len(feature_array) != len(feature_names):
                raise ValueError(f"Expected {len(feature_names)} features, got {len(feature_array)}")
            
            X_raw = np.array(feature_array).reshape(1, -1)
        
        # Formato 2: Diccionario con nombres
        else:
            print(f"[INFO] Format: Dictionary")
            missing = [f for f in feature_names if f not in input_data]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            X_raw = np.array([input_data[feat] for feat in feature_names]).reshape(1, -1)
        
        print(f"[INFO] X_raw: {X_raw}")
        
        X_norm = (X_raw - feature_mean) / (feature_std + 1e-8)
        print(f"[INFO] X_norm: {X_norm}")
        
        z = X_norm @ w + b
        probability = float(sigmoid(z)[0])
        prediction = 1 if probability >= 0.5 else 0
        
        # Clasificación de riesgo
        if probability < 0.3:
            risk_level = "BAJO"
        elif probability < 0.7:
            risk_level = "MODERADO"
        else:
            risk_level = "ALTO"
        
        result = {
            "prediction": prediction,
            "prediction_label": "Presence" if prediction == 1 else "Absence",
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "confidence": round(max(probability, 1 - probability), 4)
        }
        
        print(f"[INFO] Prediction result: {result}")
        return result
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def output_fn(prediction, accept='application/json'):
    """
    Formatea la salida.
    """
    print(f"[INFO] Formatting output with accept={accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
