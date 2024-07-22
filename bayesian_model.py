import pandas as pd
import numpy as np
from scipy import stats
import joblib


def equipment_model_inference(new_farm_data, model_path="./bayesian_model/"):
    """
    Perform inference on new farm data using the trained equipment model.

    :param new_farm_data: DataFrame containing new farm data
    :param model_path: Path where the trained models and data are saved
    :return: DataFrame of predictions for the new farm data, including cluster assignments
    """
    print("---------------- Starting inference ----------------")

    # Load saved models and data
    rf_models = joblib.load(f"{model_path}/rf_models.joblib")
    cluster_pipeline = joblib.load(f"{model_path}/cluster_model.joblib")
    preprocessor = joblib.load(f"{model_path}/preprocessor.joblib")
    farm_data_stats = joblib.load(f"{model_path}/farm_data_stats.joblib")

    # Ensure new_farm_data is a DataFrame
    if not isinstance(new_farm_data, pd.DataFrame):
        new_farm_data = pd.DataFrame([new_farm_data])
    
    # Preprocess the new data
    print("---------------- Preprocessing new data ----------------")
    required_columns = ["farm_activity_area_hectares", "main_crop_type", "well_count"]
    for col in required_columns:
        if col not in new_farm_data.columns:
            raise ValueError(f"Missing required column: {col}")
        else:
            new_farm_data[col] = pd.to_numeric(new_farm_data[col], errors='coerce')

    new_farm_data["farm_activity_area_hectares"] = new_farm_data[
        "farm_activity_area_hectares"
    ].fillna(new_farm_data["farm_activity_area_hectares"].median())
    new_farm_data["main_crop_type"] = new_farm_data["main_crop_type"].fillna(0).astype(int)
    new_farm_data["well_count"] = new_farm_data["well_count"].fillna(0).astype(int)

    # Assign clusters to new data
    print("---------------- Assigning clusters to new data ----------------")
    new_farm_data["cluster"] = cluster_pipeline.predict(new_farm_data[required_columns])

    # Generate predictions
    print("---------------- Generating predictions ----------------")
    predictions = []
    equipment_types = ["mechanical", "electrical", "submersible", "pumps"]

    for _, farm in new_farm_data.iterrows():
        farm_pred = {
            "farm_activity_area_hectares": farm["farm_activity_area_hectares"],
            "main_crop_type": farm["main_crop_type"],
            "well_count": farm["well_count"],
            "cluster": farm["cluster"],  # Include cluster assignment in predictions
        }

        # Predict equipment counts
        raw_counts = {}
        for equip_type in equipment_types:
            count_pred = rf_models[equip_type].predict(
                pd.DataFrame(
                    {
                        "farm_activity_area_hectares": [
                            farm["farm_activity_area_hectares"]
                        ],
                        "main_crop_type": [farm["main_crop_type"]],
                        "well_count": [farm["well_count"]],
                    }
                )
            )
            raw_counts[equip_type] = max(0, count_pred[0])  # Ensure non-negative

        # Adjust counts to exactly match wells_number
        total_predicted = sum(raw_counts.values())
        if total_predicted > 0:
            # Distribute wells proportionally, rounding down
            adjusted_counts = {
                equip_type: int(farm["well_count"] * count // total_predicted)
                for equip_type, count in raw_counts.items()
            }

            # Distribute any remaining wells to the equipment types with the highest fractional parts
            remaining_wells = int(farm["well_count"] - sum(adjusted_counts.values()))
            fractional_parts = {
                equip_type: farm["well_count"] * count / total_predicted
                - adjusted_counts[equip_type]
                for equip_type, count in raw_counts.items()
            }
            print(f"Remaining wells: {remaining_wells}")
            for _ in range(remaining_wells):
                equip_type = max(fractional_parts, key=fractional_parts.get)
                adjusted_counts[equip_type] += 1
                fractional_parts[equip_type] -= 1

            for equip_type in equipment_types:
                farm_pred[f"{equip_type}_equipment_count"] = adjusted_counts[equip_type]
        else:
            # If no equipment was predicted, distribute evenly
            count_per_type = farm["well_count"] // len(equipment_types)
            remainder = farm["well_count"] % len(equipment_types)
            for i, equip_type in enumerate(equipment_types):
                farm_pred[f"{equip_type}_equipment_count"] = count_per_type + (
                    1 if i < remainder else 0
                )

        # Estimate equipment sizes
        for equip_type in equipment_types:
            if farm_pred[f"{equip_type}_equipment_count"] > 0:
                stats_data = farm_data_stats[equip_type]
                prior_mean = stats_data["mean"]
                prior_std = stats_data["std"]
                data_mean = stats_data["cluster_means"].get(farm["cluster"], prior_mean)
                data_std = stats_data["cluster_stds"].get(farm["cluster"], prior_std)
                n = stats_data["cluster_counts"].get(farm["cluster"], 1)

                posterior_mean = (
                    prior_mean / prior_std**2 + data_mean * n / data_std**2
                ) / (1 / prior_std**2 + n / data_std**2)
                posterior_std = np.sqrt(1 / (1 / prior_std**2 + n / data_std**2))
                farm_pred[f"total_{equip_type}_kw"] = stats.norm(
                    posterior_mean, posterior_std
                ).rvs()
            else:
                farm_pred[f"total_{equip_type}_kw"] = 0

        predictions.append(farm_pred)

    print("---------------- Inference completed ----------------")
    print(predictions)
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
