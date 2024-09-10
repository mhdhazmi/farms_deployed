import joblib
import numpy as np
import pandas as pd
import os
from sklearn.exceptions import NotFittedError

def load_models():
    models = {}
    for model_name in ['XGBoost', 'RandomForest', 'GradientBoosting']:
        model_path = f'./saved_models/{model_name}_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[model_name] = joblib.load(model_path)
    
    preprocessor_path = './saved_models/preprocessor.pkl'
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    feature_names_path = './saved_models/feature_names.pkl'
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    feature_names = joblib.load(feature_names_path)
    
    weights_path = './saved_models/ensemble_weights.pkl'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Ensemble weights file not found: {weights_path}")
    ensemble_weights = joblib.load(weights_path)
    
    return models, preprocessor, feature_names, ensemble_weights

def check_preprocessor(preprocessor):
    """Check the state of the preprocessor and print diagnostic information."""
    print("Preprocessor diagnostics:")
    print(f"Type: {type(preprocessor)}")
    if hasattr(preprocessor, 'transformers_'):
        print("Transformers:")
        for name, transformer, columns in preprocessor.transformers_:
            print(f"  - {name}:")
            print(f"    Transformer type: {type(transformer)}")
            print(f"    Columns: {columns}")
    else:
        print("The preprocessor does not have 'transformers_' attribute.")
    print(f"Is fitted: {hasattr(preprocessor, 'n_features_in_')}")

def make_prediction(input_data):
    try:
        models, preprocessor, feature_names, ensemble_weights = load_models()
        print("------ Loaded the Models and Weights --------")
        
        # Ensure input_data has the correct features in the correct order
        missing_features = set(feature_names) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Input data is missing the following features: {missing_features}")
        
        input_data = input_data[feature_names]
        
        # Check preprocessor state
        check_preprocessor(preprocessor)
        
        # Preprocess the input data
        try:
            preprocessed_data = preprocessor.transform(input_data)
        except NotFittedError:
            raise ValueError("The preprocessor is not fitted. Please ensure it was properly fitted and saved during training.")
        
        # Make predictions with each model
        predictions = {}
        for name, model in models.items():
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                predictions[name] = model.named_steps['model'].predict(preprocessed_data)
            else:
                predictions[name] = model.predict(preprocessed_data)
        
        # Create weighted ensemble prediction
        ensemble_prediction = np.zeros_like(predictions[list(predictions.keys())[0]])
        for name, pred in predictions.items():
            ensemble_prediction += ensemble_weights[name] * pred
        
        # Add weighted ensemble prediction to the predictions dictionary
        predictions['WeightedEnsemble'] = ensemble_prediction
        
        return predictions
    
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        raise

def run_model(input_data):
    try:
        # Make predictions
        np.random.seed(42)
        random.seed(42)
        predictions = make_prediction(input_data)

        # Create a DataFrame with all predictions
        results_df = pd.DataFrame()

        for model_name, pred in predictions.items():
            results_df[f'Predicted_{model_name}'] = pred

        print("Prediction Results:")
        print(results_df)
        return results_df

    except Exception as e:
        print(f"An error occurred during model execution: {str(e)}")
        return None
