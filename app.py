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
import random
import scipy
import sklearn

debug = True


def random_fill() -> Dict[str, Any]:
    """Generate random values for farm information fields."""
    return {
        "text": f"Random text {np.random.randint(1, 100)}",
        "number": np.random.randint(0, 4),
        "float": np.random.uniform(0, 200),
    }


def main() -> None:
    """Main function to set up the Streamlit app and navigate between pages."""
    # np.random.seed(42)
    # random.seed(42)
    st.set_page_config(layout="wide", page_title="مشروع تقدير الأحمال في المزارع", page_icon="MOE_logo.png")
    st.image("MOE_logo.png", width=150)
    if "page" not in st.session_state:
        st.session_state.page = 0
    pages = [farm_info, well_info, farm_activities, summary_and_map]
    pages[st.session_state.page]()


def farm_info() -> None:
    """Display farm information input form."""
    st.title("تقدير الأحمال في المزارع")
    st.header("معلومات المزرعه")
    # random_fill_button = st.button("Fill Form with Typical Values")
    random_data = (
        random_fill() if False else {"text": "", "number": 2, "float": 100.0, "long": 24.0, "lat": 46.0, "id": 1066615101, "phone": 0599999999., "name": "أحمد", "farm_id":"1_05_20515"}
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.farm_id = st.text_input(
            "معرف المزرعة", value=random_data["farm_id"]
        )
        st.session_state.property_main_type = st.selectbox(
            "نوع المزرعة الرئيسي", list(options_new.property_main_type.keys())
        )
        st.session_state.property_area = st.number_input(
            "مساحة المزرعة (متر مربع)", min_value=0.0, value=random_data["float"]
        )
        st.session_state.well_count = st.number_input(
            "عدد الآبار", min_value=0, value=random_data["number"]
        )

    with col2:
        st.session_state.owner_name = st.text_input(
            "اسم مالك المزرعة", value=random_data["name"]
        )
        st.session_state.region = st.selectbox(
            "المنطقة", list(options_new.region.keys())
        )
        st.session_state.farm_activity_area_hectares = st.number_input(
            "مساحة الأنشطة الزراعية (هكتار)", min_value=0.0, value=random_data["float"]
        )
        st.session_state.farm_trees_count = st.number_input(
            "عدد الأشجار", min_value=0, value=random_data["number"]
        )

    with col3:
        st.session_state.national_id = st.number_input(
            "رقم أحوال مالك المزرعة", min_value=0, value=random_data["id"]
        )
        st.session_state.x_coordinate = st.number_input(
            "احداثيات خط الطول",
            min_value=-180.0,
            max_value=180.0,
       
            value=random_data["long"],
        )
        st.session_state.farm_house_count = st.number_input(
            "عدد البيوت المحمية", min_value=0, value=random_data["number"]
        )
        st.session_state.farm_plantations_count = st.number_input(
            "عدد المشاتل", min_value=0, value=random_data["number"]
        )

    with col4:
        st.session_state.phone_number = st.text_input(
            "رقم جوال المالك", value=random_data["phone"]
        )
        st.session_state.y_coordinate = st.number_input(
            "احداثيات خط العرض",
            min_value=-90.0,
            max_value=90.0,
            value=random_data["lat"],
        )
        st.session_state.activity_count = st.number_input(
            "عدد الأنشطة الزراعية", min_value=0, value=random_data["number"]
        )
        # st.session_state.farm_activity_length_m = st.number_input(
        #     "Farm Activity Length (m)", min_value=0.0, value=random_data["float"]
        # )
        # st.session_state.farm_activity_area_sq_m = st.number_input(
        #     "Farm Activity Area (m²)", min_value=0.0, value=random_data["float"]
        # )

    if st.button("Next"):
        st.session_state.page = 1
        st.experimental_rerun()


def well_info() -> None:
    """Display well information input form and store in session state."""
    st.title("معلومات الآبار")
    num_wells = st.session_state.get("well_count", 0)

    # Initialize wells dictionary if it doesn't exist
    if "wells" not in st.session_state:
        st.session_state.wells = {}

    for i in range(num_wells):
        st.subheader(f"بئر رقم {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox(
                "حالة البئر ",
                list(options_new.well_is_active.keys()),
                key=f"well_is_active_{i}",
            )
            irrigation_type = st.selectbox(
                "نوع ري البئر ",
                list(options_new.well_irrigation_type.keys()),
                key=f"well_irrigation_type_{i}",
            )
        with col2:
            possession_type = st.selectbox(
                "نوع ملكية البئر ",
                ["مملوك"],
                key=f"well_possession_type_{i}",
            )
            irrigation_source = st.selectbox(
                "مصدر مياه البئر ",
                list(options_new.well_irrigation_source.keys()),
                key=f"well_irrigation_source_{i}",
            )

        # Store well information in session state
        st.session_state.wells[i] = {
            "well_is_active": status,
            "irrigation_type": irrigation_type,
            "possession_type": possession_type,
            "irrigation_source": irrigation_source,
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
    st.title("معلومات الأنشطة الزراعية")
    num_activities = st.session_state.get("activity_count", 0)
    for i in range(num_activities):
        st.subheader(f"النشاط رقم {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            # was deleted because of conflict with the first page
            # st.number_input(
            #     "مساحة النشاط (هكتار)",
            #     min_value=0.0,
            #     step=0.1,
            #     key=f"farm_activity_area_hectares_{i}",
            # )
            st.selectbox(
                "نوع المحصول الأساسي للنشاط",
                list(options_new.farm_main_crops_type.keys()),
                key=f"farm_main_crops_type_{i}",
            )
            st.selectbox(
                "حالة النشاط",
                list(options_new.farm_activity_status.keys()),
                key=f"farm_activity_status_{i}",
            )
        with col2:
            st.selectbox(
                "نوع النشاط",
                list(options_new.farm_type.keys()),
                key=f"farm_type_{i}",
            )
            st.selectbox(
                "الموسم الزراعي",
                list(options_new.farm_farming_season.keys()),
                key=f"farm_farming_season_{i}",
            )
            st.selectbox(
                "مصدر الري للنشاط",
                list(options_new.farm_irrigation_source.keys()),
                key=f"farm_irrigation_source_{i}",
            )
            st.selectbox(
                "نوع الري للنشاط",
                list(options_new.farm_irrigation_type.keys()),
                key=f"farm_irrigation_type_{i}",
            )

    st.header("معلومات البيوت المحمية")
    num_houses = st.session_state.get("farm_house_count", 0)
    for i in range(num_houses):
        st.selectbox(
            f"نوع البيت المحمي رقم {i + 1}",
            list(options_new.farm_house_type.keys()),
            key=f"farm_house_type_{i}",
        )

    st.header("معلومات المشاتل")
    num_plantations = st.session_state.get("farm_plantations_count", 0)
    for i in range(num_plantations):
        st.selectbox(
            f"معلومات المشتل رقم {i + 1}",
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
            predictions = bayesian_model.improved_equipment_model_inference(prediction)
            prediction = pd.concat([prediction, predictions], axis=1)
            prediction = prediction[[
                # 'city',
    'wells_number',
    'mechanical_equipment_count',
    'electrical_equipment_count',
    'submersible_equipment_count',
    'pumps_equipment_count',
    'total_mechanical_kw',
    'total_electrical_kw',
    'total_submersible_kw',
    'total_pumps_kw',
    'cluster',
    'sprinklers_equipment_kw',
    'property_area',
    'property_main_type',
    'well_count',
    'farm_trees_count',
    'farm_house_count',
    'farm_plantations_count',
    'main_crop_type',
    'well_possession_type_1',
    # 'well_possession_type_2',
    # 'well_is_active_0',
    'well_is_active_1',
    'well_is_active_2',
    # 'well_irrigation_source_0',
    'well_irrigation_source_1',
    'well_irrigation_source_2',
    'well_irrigation_source_4',
    'well_irrigation_source_5',
    'well_irrigation_source_6',
    'well_irrigation_source_10',
    'well_irrigation_source_12',
    # 'well_irrigation_type_0.0',
    'well_irrigation_type_1',
    'well_irrigation_type_2',
    'well_irrigation_type_3',
    'well_irrigation_type_4',
    'well_irrigation_type_6',
    'well_irrigation_type_7',
    'farm_activity_area_hectares',
    # 'farm_irrigation_source_0.0',
    'farm_irrigation_source_1',
    'farm_irrigation_source_2',
    'farm_irrigation_source_4',
    'farm_irrigation_source_5',
    'farm_irrigation_source_6',
    'farm_irrigation_source_7',
    'farm_irrigation_source_10',
    'farm_irrigation_source_12',
    # 'farm_irrigation_type_0.0',
    'farm_irrigation_type_1',
    'farm_irrigation_type_2',
    'farm_irrigation_type_3',
    'farm_irrigation_type_4',
    'farm_irrigation_type_6',
    'farm_irrigation_type_7',
    # 'farm_activity_length_m',
    'farm_activity_area_sq_m',
    'farm_geometry',
    # 'farm_main_crops_type_0.0',
    'farm_main_crops_type_1',
    'farm_main_crops_type_2',
    'farm_main_crops_type_3',
    'farm_main_crops_type_4',
    'farm_main_crops_type_5',
    'farm_main_crops_type_6',
    'farm_main_crops_type_7',
    'farm_main_crops_type_8',
    'farm_main_crops_type_9',
    'farm_main_crops_type_10',
    'farm_main_crops_type_12',
    'farm_main_crops_type_13',
    'farm_main_crops_type_14',
    'farm_main_crops_type_15',
    # 'farm_activity_status_0',
    'farm_activity_status_1',
    'farm_activity_status_3',
    'farm_activity_status_4',
    'farm_activity_status_6',
    # 'farm_type_0.0',
    'farm_type_1',
    'farm_type_2',
    'farm_type_5',
    'farm_type_6',
    'farm_type_7',
    'farm_type_10',
    'farm_type_11',
    # 'farm_farming_season_0.0',
    'farm_farming_season_1',
    'farm_farming_season_2',
    'farm_farming_season_3',
    'farm_farming_season_4',
    # 'farm_house_type_0.0',
    'farm_house_type_1',
    'farm_house_type_2',
    'farm_house_type_3',
    'farm_house_type_4',
    'farm_house_type_6',
    'farm_house_type_7',
    # 'farm_plantations_type_0.0',
    'farm_plantations_type_1',
    'farm_plantations_type_2',
    'farm_plantations_type_3'
            ]]
            prediction.to_csv("prediction.csv", index=False)
            print(
                "------------------------ Data saved to drive ----------------------------"
            )
            if debug:
                prediction = pd.read_csv("./test_data.csv")
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
        # "property_main_type",
        "farm_type",
        "farm_main_crops_type",
        "farm_plantations_type",
        "farm_house_type",
        # "region",
        "farm_farming_season",
        "farm_activity_status",
        "farm_irrigation_source",
        "farm_irrigation_type",
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


def update_categorical(
    df: pd.DataFrame, field: str, value: Any, count: int = 1
) -> None:
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
        "wells_number",
        "well_count",
        "farm_trees_count",
        "farm_house_count",
        "farm_plantations_count",
        "farm_activity_area_hectares",
    ]
    for field in numeric_fields:
        df.at[0, field] = inputs.get(field, 0)
    df["wells_number"] = df["well_count"].values

    df["farm_geometry"] = inputs.get("activity_count", 0)
    df["sprinklers_equipment_kw"] = 25 * inputs.get("well_count", 0)
    df["property_area"] = st.session_state.property_area
    df["farm_activity_area_sq_m"] = inputs.get("farm_activity_area_hectares", 0) * 10000
    # st.session_state.farm_activity_area_hectares * 10000
    df["property_main_type"] = getattr(options_new, "property_main_type").get(inputs.get("property_main_type", 0))
    ################## Testing ##############################
    # df['farm_activity_area_hectares'] = 165.999277
    df["main_crop_type"] = 1 # options_new.farm_main_crops_type[inputs.get("farm_main_type", 0)]
    #df["farm_crops_type"] = 1


    ########################################################

    # Update categorical fields
    categorical_fields = [
        "property_main_type",
        "farm_type",
        "farm_main_crops_type",
        "farm_plantations_type",
        "farm_house_type",
        "region",
        "farm_farming_season",
        "farm_activity_status",
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
        df.at[0, "farm_activity_area_hectares"] += inputs.get(
            f"farm_activity_area_hectares_{i}", 0
        )
        activity_fields = [
            "farm_main_crops_type",
            "farm_activity_status",
            "farm_farming_season",
            "farm_irrigation_source",
            "farm_irrigation_type",
            "farm_type",
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
        update_categorical(
            df, "farm_plantations_type", inputs.get(f"farm_plantations_type_{i}")
        )

    columns = [f'farm_main_crops_type_{i}' for i in range(1, 16)]
    # print(df[columns])
    df['main_crop_type'] = df[columns].apply(lambda row: row[row != 0].nunique(), axis=1)

    return df


def summary_and_map() -> None:
    """Display summary and map based on inputs."""
    st.title("تقدير الأحمال في المزارع")
    # st.header("ملخص المزرعة")
    # st.write(f"Farm ID: {st.session_state.farm_id}")
    # st.write(f"مالك المزرعة: {st.session_state.owner_name}")
    # st.write(f"مساحة المزرعة: {st.session_state.property_area} m²")
    # st.write(f"نوع المزرعة: {st.session_state.property_main_type}")
    # st.write(f"عدد الآبار: {st.session_state.well_count}")
    # st.write(f"عدد النشاطات: {st.session_state.activity_count}")

    st.session_state.model_results.to_dict()
    st.header("نتائج النموذج الذكي")
    # st.json(st.session_state.model_results.to_dict())
    # st.json(st.session_state.prediction.to_dict())
    # st.write(
    #     f'عدد الأجهزة الميكانيكية التي تخدم الآبار هي {list(st.session_state.prediction.to_dict().get("mechanical_equipment_count").values())[0]} وحملها {list(st.session_state.prediction.to_dict().get("total_mechanical_kw").values())[0]:.2f} كيلوواط'
    # )
    # st.write(
    #     f'عدد الأجهزة الكهربائية التي تخدم الآبار هي {list(st.session_state.prediction.to_dict().get("electrical_equipment_count").values())[0]} وحملها {list(st.session_state.prediction.to_dict().get("total_electrical_kw").values())[0]:.2f} كيلوواط'
    # )
    # st.write(
    #     f'عدد الغطاسات التي تخدم الآبار هي {list(st.session_state.prediction.to_dict().get("submersible_equipment_count").values())[0]} وحملها {list(st.session_state.prediction.to_dict().get("total_submersible_kw").values())[0]:.2f} كيلوواط'
    # )
    # st.write(
    #     f'عدد المضخات التي تخدم الآبار هي {list(st.session_state.prediction.to_dict().get("pumps_equipment_count").values())[0]} وحملها {list(st.session_state.prediction.to_dict().get("total_pumps_kw").values())[0]:.2f} كيلوواط'
    # )
    st.subheader("الأجهزة المستخدمه في الآبار")
    data = {
        'نوع الجهاز': ['رأس ميكانيكي', 'رأس كهربائي', 'الغطاسات', 'المضخات'],
        'العدد': [
            int(list(st.session_state.prediction.to_dict().get("mechanical_equipment_count").values())[0]),
            int(list(st.session_state.prediction.to_dict().get("electrical_equipment_count").values())[0]),
            int(list(st.session_state.prediction.to_dict().get("submersible_equipment_count").values())[0]),
            int(list(st.session_state.prediction.to_dict().get("pumps_equipment_count").values())[0])
        ],
        'الحمل (كيلوواط)': [
            f"{list(st.session_state.prediction.to_dict().get('total_mechanical_kw').values())[0]:.2f}",
            f"{list(st.session_state.prediction.to_dict().get('total_electrical_kw').values())[0]:.2f}",
            f"{list(st.session_state.prediction.to_dict().get('total_submersible_kw').values())[0]:.2f}",
            f"{list(st.session_state.prediction.to_dict().get('total_pumps_kw').values())[0]:.2f}"
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)


    # Display the table
    st.table(df)
    
    models_predictions = list(st.session_state.model_results.to_dict().values())
    results_list = [list(d.values())[0] for d in models_predictions]
    
    # st.subheader("استهلاك المزرعة")
    # st.subheader(f" الاستهلاك الأدنى للمزرعة هو {min(results_list):.2f} كيلوواط")
    # st.subheader(f"  الاستهلاك الأعلى للمزرعة هو {max(results_list):.2f} كيلوواط")
    
    min_consumption = min(results_list)
    max_consumption = max(results_list)
    ideal_consumption = results_list[3]

    st.header("استهلاك المزرعة", divider="rainbow")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="الاستهلاك الأدنى للمزرعة",
            value=f"{min_consumption:.2f} كيلوواط",
            delta=f"{min_consumption - ideal_consumption:.2f} كيلوواط",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="الاستهلاك الأعلى للمزرعة",
            value=f"{max_consumption:.2f} كيلوواط",
            delta=f"{max_consumption - ideal_consumption:.2f} كيلوواط",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="الاستهلاك المتوقع للمزرعة",
            value=f"{ideal_consumption:.2f} كيلوواط",
            delta=None
        )


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
