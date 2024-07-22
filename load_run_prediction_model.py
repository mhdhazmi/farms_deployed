import joblib
import numpy as np
import pandas as pd

def load_models():
    models = {}
    for model_name in ['XGBoost', 'RandomForest', 'GradientBoosting']:
        models[model_name] = joblib.load(f'./saved_models/{model_name}_model.pkl')
    
    preprocessor = joblib.load('./saved_models/preprocessor.pkl')
    feature_names = joblib.load('./saved_models/feature_names.pkl')
    return models, preprocessor, feature_names

def make_prediction(input_data):
    models, preprocessor, feature_names = load_models()
    print("------ Loaded the Models --------")
    
    # Ensure input_data has the correct features in the correct order
    input_data = input_data[feature_names]
    
    # Make predictions with each model
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_data)
        
    
    # Create ensemble prediction (simple average)
    ensemble_prediction = np.mean(list(predictions.values()), axis=0)
    
    
    # Add ensemble prediction to the predictions dictionary
    predictions['Ensemble'] = ensemble_prediction
    
    return predictions

def run_model(final_df_):
    # Make predictions
    data = final_df_.copy()
    predictions = make_prediction(data)

    # Create a DataFrame with all predictions and actual values
    results_df = pd.DataFrame({
        #'Actual': final_df_['total_electrical_load_kw'],
        
    })

    for model_name, pred in predictions.items():
        results_df[f'Predicted_{model_name}'] = pred
    print(results_df)
    return results_df