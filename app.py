import json
from typing import Any, Dict

# from matplotlib.pyplot import plot
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import options_new
import bayesian_model
import joblib
import load_run_prediction_model as lr
import scipy
import sklearn


def random_fill() -> Dict[str, Any]:
    """Generate random values for farm information fields."""
    return {
        "text": f"Random text {np.random.randint(1, 100)}",
        "number": np.random.randint(0, 4),
        "float": np.random.uniform(0, 1000000000),
    }


def main() -> None:

    """Main function to set up the Streamlit app and navigate between pages."""
    st.set_page_config(layout="wide")
    st.image("MOE_logo.png", width=150)
    if "page" not in st.session_state:
        st.session_state.page = 0
    pages = [farm_info, well_info, farm_activities, summary_and_map]
    pages[st.session_state.page]()


def farm_info() -> None:
    """Display farm information input form."""
    st.title("معلومات المزرعة")
    st.header("Farm Details")
    random_fill_button = st.button("Fill Form with Typical Values")
    random_data = random_fill() if random_fill_button else {"text": "", "number": 2, "float": 0.0}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.farm_id = st.text_input("Farm ID", value=random_data["text"])
        st.session_state.property_main_type = st.selectbox(
            "Property Main Type", list(options_new.property_main_type.keys())
        )
        st.session_state.property_area = st.number_input(
            "Property Area (m²)", min_value=0.0, value=random_data["float"]
        )
        st.session_state.well_count = st.number_input(
            "Number of Wells", min_value=0, value=random_data["number"]
        )
        st.session_state.activity_count = st.number_input(
            "Number of Activities", min_value=0, value=random_data["number"]
        )

    with col2:
        st.session_state.owner_name = st.text_input("Farm Owner Name", value=random_data["text"])
        st.session_state.region = st.selectbox("Region", list(options_new.region.keys()))
        st.session_state.farm_activity_area_hectares = st.number_input(
            "Farm Activity Area (hectares)", min_value=0.0, value=random_data["float"]
        )
        st.session_state.farm_trees_count = st.number_input(
            "Number of Trees", min_value=0, value=random_data["number"]
        )

    with col3:
        st.session_state.national_id = st.number_input("Owner National ID", min_value=0, value=random_data["number"])
        st.session_state.x_coordinate = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=random_data["float"]
        )
        st.session_state.farm_house_count = st.number_input(
            "Number of Houses", min_value=0, value=random_data["number"]
        )
        st.session_state.farm_plantations_count = st.number_input(
            "Number of Plantations", min_value=0, value=random_data["number"]
        )

    with col4:
        st.session_state.phone_number = st.text_input("Phone Number", value=random_data["text"])
        st.session_state.y_coordinate = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0, value=random_data["float"]
        )
        st.session_state.farm_activity_length_m = st.number_input(
            "Farm Activity Length (m)", min_value=0.0, value=random_data["float"]
        )
        st.session_state.farm_activity_area_sq_m = st.number_input(
            "Farm Activity Area (m²)", min_value=0.0, value=random_data["float"]
        )

    if st.button("Next"):
        st.session_state.page = 1
        st.experimental_rerun()


def well_info() -> None:
    """Display well information input form and store in session state."""
    st.title("Well Information")
    num_wells = st.session_state.get("well_count", 0)
    
    # Initialize wells dictionary if it doesn't exist
    if "wells" not in st.session_state:
        st.session_state.wells = {}

    for i in range(num_wells):
        st.subheader(f"Well {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox(
                f"Status of Well {i + 1}",
                list(options_new.well_is_active.keys()),
                key=f"well_is_active_{i}",
            )
            irrigation_type = st.selectbox(
                f"Type of Well Irrigation {i + 1}",
                list(options_new.well_irrigation_type.keys()),
                key=f"well_irrigation_type_{i}",
            )
        with col2:
            possession_type = st.selectbox(
                f"Possession Type of Well {i + 1}",
                ["Owned"],
                key=f"well_possession_type_{i}",
            )
            irrigation_source = st.selectbox(
                f"Source of Water for Well {i + 1}",
                list(options_new.well_irrigation_source.keys()),
                key=f"well_irrigation_source_{i}",
            )
        
        # Store well information in session state
        st.session_state.wells[i] = {
            "well_is_active": status,
            "irrigation_type": irrigation_type,
            "possession_type": possession_type,
            "irrigation_source": irrigation_source
        }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 0
            st.experimental_rerun()
    with col2:
        if st.button("Next"):
            st.session_state.page = 2
            st.experimental_rerun()


def farm_activities() -> None:
    """Display farm activities input form."""
    st.title("Farm Activities Information")
    num_activities = st.session_state.get("activity_count", 0)
    for i in range(num_activities):
        st.subheader(f"Activity {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                f"Activity {i + 1} Area (hectares)",
                min_value=0.0,
                step=0.1,
                key=f"farm_activity_area_hectares_{i}",
            )
            st.selectbox(
                f"Main Crop Type for Activity {i + 1}",
                list(options_new.farm_main_crops_type.keys()),
                key=f"farm_main_crops_type_{i}",
            )
            st.selectbox(
                f"Activity {i + 1} Status",
                list(options_new.farm_activity_status.keys()),
                key=f"farm_activity_status_{i}",
            )
        with col2:
            st.selectbox(
                f"Type of Activity {i + 1}",
                list(options_new.farm_type.keys()),
                key=f"farm_type_{i}",
            )
            st.selectbox(
                f"Farming Season for Activity {i + 1}",
                list(options_new.farm_farming_season.keys()),
                key=f"farm_farming_season_{i}",
            )
            st.selectbox(
                f"Irrigation Source(s) for Activity {i + 1}",
                list(options_new.farm_irrigation_source.keys()),
                key=f"farm_irrigation_source_{i}",
            )
            st.selectbox(
                f"Irrigation Type(s) for Activity {i + 1}",
                list(options_new.farm_irrigation_type.keys()),
                key=f"farm_irrigation_type_{i}",
            )

    st.header("Farm Houses Information")
    num_houses = st.session_state.get("farm_house_count", 0)
    for i in range(num_houses):
        st.selectbox(
            f"Type of Farm House {i + 1}",
            list(options_new.farm_house_type.keys()),
            key=f"farm_house_type_{i}",
        )

    st.header("Farm Plantations Information")
    num_plantations = st.session_state.get("farm_plantations_count", 0)
    for i in range(num_plantations):
        st.selectbox(
            f"Type of Farm Plantation {i + 1}",
            list(options_new.farm_plantations_type.keys()),
            key=f"farm_plantations_type_{i}",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 1
            st.experimental_rerun()
    with col2:
        if st.button("Next"):
            prediction = predict(st.session_state)
            print(prediction['well_count'])
            predictions = bayesian_model.equipment_model_inference(prediction)
            prediction = pd.concat([prediction,predictions], axis=1)
            print(prediction['well_count'])
            prediction.to_csv("prediction.csv", index=False)
            print("------------------------ Data saved to drive ----------------------------")
            model_results = lr.run_model(prediction)
            st.session_state.prediction = prediction
            st.session_state.model_results = model_results
            st.session_state.page = 3
            st.experimental_rerun()


def create_dataframe_from_options() -> pd.DataFrame:
    """Create an empty DataFrame with columns based on options."""
    columns = [
        "wells_number",
        "well_count",
        "farm_trees_count",
        "farm_house_count",
        "farm_plantations_count",
        "farm_activity_area_hectares",
        "region",
        "property_main_type",
    ]

    relevant_dicts = [
        "well_irrigation_source",
        "well_irrigation_type",
        "well_is_active",
        "well_possession_type",
        #"property_main_type",
        "farm_type",
        "farm_main_crops_type",
        "farm_plantations_type",
        "farm_house_type",
        # "region",
        "farm_farming_season",
        "farm_activity_status",
        "farm_irrigation_source",
        "farm_irrigation_type"
    ]

    for dict_name in relevant_dicts:
        # print(dict_name)
        if hasattr(options_new, dict_name):
            # print(dict_name)
            option_dict = getattr(options_new, dict_name)
            if isinstance(option_dict, dict):
                for value in option_dict.values():
                    # print(f"Here is the dictionary and value {dict_name}_{value}")
                    columns.append(f"{dict_name}_{value}")

    df = pd.DataFrame(0, index=[0], columns=columns)
    return df


def update_categorical(df: pd.DataFrame, field: str, value: Any, count: int = 1) -> None:
    """Update categorical fields in the DataFrame."""
    # print(field, value)
    if value:
        if isinstance(value, list):
            for v in value:
                col_name = f"{field}_{getattr(options_new, field).get(v, '')}"
                if col_name in df.columns:
                    # print(f"Here is the column name: {col_name}")
                    df.at[0, col_name] += count
        elif isinstance(value, str):
            col_name = f"{field}_{getattr(options_new, field).get(value, '')}"
            if col_name in df.columns:
                df.at[0, col_name] += count


def predict(inputs: Dict[str, Any]) -> pd.DataFrame:
    """Generate predictions based on inputs."""
    df = create_dataframe_from_options()

    # Update basic information
    numeric_fields = [
        "wells_number", "well_count", "farm_trees_count", "farm_house_count",
        "farm_plantations_count", "farm_activity_area_hectares"
    ]
    for field in numeric_fields:
        df.at[0, field] = inputs.get(field, 0)
    df['wells_number'] = df['well_count'].values
    
    df['farm_geometry'] = inputs.get("activity_count", 0)
    df['sprinklers_equipment_kw'] = 25*inputs.get("well_count", 0)
    ################## Testing ##############################
    # df['farm_activity_area_hectares'] = 165.999277
    df['main_crop_type'] = 3 # options_new.farm_main_crops_type[inputs.get("farm_main_crops_type", 0)]
    df['farm_crops_type'] = 3 # options_new.farm_main_crops_type[inputs.get("farm_main_crops_type", 0)] #inputs.get("farm_main_crop_type", 0)
    
    df['farm_activity_area_sq_m'] = 6579.811336	
    df['property_area'] = 6579.811336	
    
    ########################################################

    # Update categorical fields
    categorical_fields = [
        "property_main_type", "farm_type", "farm_main_crops_type",
        "farm_plantations_type", "farm_house_type", "region", 
        "farm_farming_season", "farm_activity_status"
    ]
    for field in categorical_fields:
        # print(field, inputs.get(field))
        update_categorical(df, field, inputs.get(field))

    # Process wells
    wells = inputs.get("wells", {})
    for well_data in wells.values():
        update_categorical(df, "well_irrigation_source", well_data["irrigation_source"])
        update_categorical(df, "well_irrigation_type", well_data["irrigation_type"])
        update_categorical(df, "well_possession_type", well_data["possession_type"])
        update_categorical(df, "well_is_active", well_data["well_is_active"])

    # Process farm activities
    activity_count = inputs.get("activity_count", 0)
    for i in range(activity_count):
        df.at[0, "farm_activity_area_hectares"] += inputs.get(f"farm_activity_area_hectares_{i}", 0)
        activity_fields = [
            "farm_main_crops_type", "farm_activity_status", "farm_farming_season",
            "farm_irrigation_source", "farm_irrigation_type", "farm_type"
        ]
        for field in activity_fields:
            update_categorical(df, field, inputs.get(f"{field}_{i}"))

    # Process farm houses
    house_count = inputs.get("farm_house_count", 0)
    for i in range(house_count):
        update_categorical(df, "farm_house_type", inputs.get(f"farm_house_type_{i}"))

    # Process farm plantations
    plantation_count = inputs.get("farm_plantations_count", 0)
    for i in range(plantation_count):
        update_categorical(df, "farm_plantations_type", inputs.get(f"farm_plantations_type_{i}"))
    

    return df


def summary_and_map() -> None:
    """Display summary and map based on inputs."""
    st.title("Summary and Map")
    st.header("Farm Summary")
    st.write(f"Farm ID: {st.session_state.farm_id}")
    st.write(f"Owner: {st.session_state.owner_name}")
    st.write(f"Property Area: {st.session_state.property_area} m²")
    st.write(f"Farm Main Type: {st.session_state.property_main_type}")
    st.write(f"Number of Wells: {st.session_state.well_count}")
    st.write(f"Number of Activities: {st.session_state.activity_count}")
    

    st.session_state.model_results.to_dict()
    st.header("Prediction Results")
    st.json(st.session_state.model_results.to_dict())

    map_data = pd.DataFrame(
        {"lat": [st.session_state.y_coordinate], "lon": [st.session_state.x_coordinate]}
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=pdk.ViewState(
                latitude=st.session_state.y_coordinate,
                longitude=st.session_state.x_coordinate,
                zoom=12,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0, 160]",
                    get_radius=200,
                ),
            ],
        )
    )

    if st.button("Previous"):
        st.session_state.page = 2
        st.experimental_rerun()




if __name__ == "__main__":
    main()
