import pandas as pd
import numpy as np
from scipy import stats
import joblib

def improved_equipment_model_inference(new_farm_data, model_path="./bayesian_model/"):
    """
    Perform inference on new farm data to estimate equipment counts and capacities.
    
    :param new_farm_data: DataFrame containing new farm data
    :param model_path: Path where the trained models and data are saved
    :return: DataFrame of predictions for equipment counts and capacities
    """
    print('---------------- Starting improved inference ----------------')

    try:
        # Load saved models and data
        rf_models = joblib.load(f'{model_path}/rf_models.joblib')
        cluster_pipeline = joblib.load(f'{model_path}/cluster_model.joblib')
        farm_data_stats = joblib.load(f'{model_path}/farm_data_stats.joblib')

        # Ensure new_farm_data is a DataFrame
        if not isinstance(new_farm_data, pd.DataFrame):
            new_farm_data = pd.DataFrame([new_farm_data])

        # Check for required columns
        required_columns = ['farm_activity_area_hectares', 'main_crop_type', 'well_count']
        missing_columns = set(required_columns) - set(new_farm_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Preprocess the new data
        print('---------------- Preprocessing new data ----------------')
        new_farm_data['farm_activity_area_hectares'] = new_farm_data['farm_activity_area_hectares'].fillna(new_farm_data['farm_activity_area_hectares'].median())
        new_farm_data['main_crop_type'] = new_farm_data['main_crop_type'].fillna(-1).astype(int)
        new_farm_data['well_count'] = new_farm_data['well_count'].fillna(0).astype(int)  # Ensure well_count is integer

        # Assign clusters to new data
        print('---------------- Assigning clusters to new data ----------------')
        new_farm_data['cluster'] = cluster_pipeline.predict(new_farm_data[required_columns])

        # Generate predictions
        print('---------------- Generating improved predictions ----------------')
        predictions = []
        equipment_types = ['mechanical', 'electrical', 'submersible', 'pumps']
        
        for _, farm in new_farm_data.iterrows():
            farm_pred = {
                'farm_activity_area_hectares': farm['farm_activity_area_hectares'],
                'main_crop_type': farm['main_crop_type'],
                'well_count': int(farm['well_count']),  # Ensure well_count is integer
                'cluster': farm['cluster']
            }
            
            # Predict equipment counts
            raw_counts = {}
            for equip_type in equipment_types:
                count_pred = rf_models[equip_type].predict(pd.DataFrame({
                    'farm_activity_area_hectares': [farm['farm_activity_area_hectares']],
                    'main_crop_type': [farm['main_crop_type']],
                    'well_count': [farm['well_count']]
                }))
                raw_counts[equip_type] = max(0, count_pred[0])  # Ensure non-negative
            
            # Adjust counts to exactly match wells_number
            total_predicted = sum(raw_counts.values())
            if total_predicted > 0:
                adjusted_counts = {equip_type: int(farm_pred['well_count'] * count / total_predicted) 
                                   for equip_type, count in raw_counts.items()}
                remaining_wells = farm_pred['well_count'] - sum(adjusted_counts.values())
                fractional_parts = {equip_type: farm_pred['well_count'] * count / total_predicted - adjusted_counts[equip_type] 
                                    for equip_type, count in raw_counts.items()}
                
                # Debug output
                print(f"Debug - well_count: {farm_pred['well_count']}, type: {type(farm_pred['well_count'])}")
                print(f"Debug - remaining_wells: {remaining_wells}, type: {type(remaining_wells)}")
                
                for _ in range(int(remaining_wells)):  # Ensure remaining_wells is integer
                    equip_type = max(fractional_parts, key=fractional_parts.get)
                    adjusted_counts[equip_type] += 1
                    fractional_parts[equip_type] -= 1
                
                for equip_type in equipment_types:
                    farm_pred[f'{equip_type}_equipment_count'] = adjusted_counts[equip_type]
            else:
                count_per_type = farm_pred['well_count'] // len(equipment_types)
                remainder = farm_pred['well_count'] % len(equipment_types)
                for i, equip_type in enumerate(equipment_types):
                    farm_pred[f'{equip_type}_equipment_count'] = count_per_type + (1 if i < remainder else 0)
            
            # Improved equipment size estimation
            for equip_type in equipment_types:
                equipment_count = farm_pred[f'{equip_type}_equipment_count']
                if equipment_count > 0:
                    stats_data = farm_data_stats[equip_type]
                    prior_mean = stats_data['mean']
                    prior_std = stats_data['std']
                    data_mean = stats_data['cluster_means'].get(farm_pred['cluster'], prior_mean)
                    data_std = stats_data['cluster_stds'].get(farm_pred['cluster'], prior_std)
                    n = stats_data['cluster_counts'].get(farm_pred['cluster'], 1)
                    
                    posterior_mean = (prior_mean/prior_std**2 + data_mean*n/data_std**2) / (1/prior_std**2 + n/data_std**2)
                    posterior_std = np.sqrt(1 / (1/prior_std**2 + n/data_std**2))
                    
                    # Generate individual equipment sizes
                    individual_sizes = stats.truncnorm(
                        a=(0 - posterior_mean) / posterior_std,
                        b=np.inf,
                        loc=posterior_mean,
                        scale=posterior_std
                    ).rvs(size=equipment_count)
                    
                    farm_pred[f'total_{equip_type}_kw'] = np.sum(individual_sizes)
                else:
                    farm_pred[f'total_{equip_type}_kw'] = 0
            
            predictions.append(farm_pred)

        print('---------------- Improved inference completed ----------------')
        return pd.DataFrame(predictions)[
            [
                "mechanical_equipment_count",
                "electrical_equipment_count",
                "submersible_equipment_count",
                "pumps_equipment_count",
                "total_mechanical_kw",
                "total_electrical_kw",
                "total_submersible_kw",
                "total_pumps_kw",
            ]
        ]

    except Exception as e:
        print(f"An error occurred during inference: {str(e)}")
        raise