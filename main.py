import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import plotly.graph_objects as go

def get_clean_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
    data = data.drop(["Unnamed: 32",'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return data

def get_scaled_value(input_data):
    data = get_clean_data()
    x = data.drop(["diagnosis"], axis=1)
    scaled_dict = {}
    for key, value in input_data.items():
        min_value = x[key].min() #min value in column
        max_value = x[key].max() #max in column
        scaled_value = (value - min_value) / (max_value - min_value) #scaling formula
        scaled_dict[key] = scaled_value
    return scaled_dict
def create_chart(input_data):
    input_data = get_scaled_value(input_data) #scaling the data to plot the graph bw 0 and 1

    # Features (axes of the radar chart)
    features = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness','Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']



    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'],input_data['texture_mean'],input_data['perimeter_mean'],input_data['area_mean'],input_data['smoothness_mean'],input_data['compactness_mean'],input_data['concavity_mean'],input_data['concave points_mean'],input_data['symmetry_mean'],input_data['fractal_dimension_mean']],

        theta=features,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'],input_data['texture_se'],input_data['perimeter_se'],input_data['area_se'],input_data['smoothness_se'],input_data['compactness_se'],input_data['concavity_se'],input_data['concave points_se'],input_data['symmetry_se'],input_data['fractal_dimension_se']],
        theta=features,

        fill='toself',
        name='SE Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [input_data['radius_worst'],input_data['texture_worst'],input_data['perimeter_worst'],input_data['area_worst'],input_data['smoothness_worst'],input_data['compactness_worst'],input_data['concavity_worst'],input_data['concave points_worst'],input_data['symmetry_worst'],input_data['fractal_dimension_worst']],
        theta=features,

        fill='toself',
        name = 'Worst Value'


    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def create_side_bar():


    # Step 1: Read your dataset
    df = get_clean_data() # change path to your dataset

    # Step 2: Column name → Friendly label mapping
    feature_labels = {
        "radius_mean": "Mean Radius",
        "texture_mean": "Mean Texture",
        "perimeter_mean": "Mean Perimeter",
        "area_mean": "Mean Area",
        "smoothness_mean": "Mean Smoothness",
        "compactness_mean": "Mean Compactness",
        "concavity_mean": "Mean Concavity",
        "concave points_mean": "Mean Concave Points",
        "symmetry_mean": "Mean Symmetry",
        "fractal_dimension_mean": "Mean Fractal Dimension",
        "radius_se": "Radius SE",
        "texture_se": "Texture SE",
        "perimeter_se": "Perimeter SE",
        "area_se": "Area SE",
        "smoothness_se": "Smoothness SE",
        "compactness_se": "Compactness SE",
        "concavity_se": "Concavity SE",
        "concave points_se": "Concave Points SE",
        "symmetry_se": "Symmetry SE",
        "fractal_dimension_se": "Fractal Dimension SE",
        "radius_worst": "Worst Radius",
        "texture_worst": "Worst Texture",
        "perimeter_worst": "Worst Perimeter",
        "area_worst": "Worst Area",
        "smoothness_worst": "Worst Smoothness",
        "compactness_worst": "Worst Compactness",
        "concavity_worst": "Worst Concavity",
        "concave points_worst": "Worst Concave Points",
        "symmetry_worst": "Worst Symmetry",
        "fractal_dimension_worst": "Worst Fractal Dimension"
    }

    # Step 3: Create sliders with dynamic ranges
    st.sidebar.title("Breast Cancer Feature Input")

    input_dict = {}
    for col, label in feature_labels.items():
        min_val = 0.0
        max_val = round(float(df[col].max()),2)
        default_val = float(df[col].mean())
        step_val = round((max_val - min_val) / 100, 4)  # dynamic step size

        input_dict[col] =st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val
        )

    return input_dict
def colored_box(text, color, text_color="White"):
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 12px;
            border-radius: 8px;
            color: {text_color};
            font-weight: bold;
            text-align: center;">
            {text}
        </div>
        """,
        unsafe_allow_html=True)

def add_prediction(input_value):
    model = pickle.load(open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(os.path.dirname(__file__), "scaler.pkl"), "rb"))

    input_array = np.array(list(input_value.values())).reshape(1, -1)
    scaled = scaler.transform(input_array)
    prediction = model.predict(scaled)
    st.subheader("Cell Prediction")
    st.write("The Cell Cluster is: ")
    if prediction == 1:
        colored_box("Malignant", "red")
    else:
        colored_box("Benign", "green")

    st.write("Probability of being Benign: ",round(model.predict_proba(scaled)[0][0],4))
    st.write("Probability of being Malignant: ", round(model.predict_proba(scaled)[0][1],4))


def main():
    st.set_page_config(
        page_title = 'Breast Cancer Prediction App',
        page_icon = 'female-doctor',
        layout = 'wide',
        initial_sidebar_state = 'expanded'
    )
    input_data = create_side_bar()

    with st.container():
        st.title("Breast Cancer Prediction App")
        st.write("This app lets you adjust breast cancer test measurements using simple sliders. It uses real medical data ranges and predicts whether the cells are malignant or benign, making it useful for learning, testing, or healthcare support.")

    col1, col2=st.columns([4,1], border = True)
    with col1:
        chart = create_chart(input_data)
        st.plotly_chart(chart)
    with col2:

        add_prediction(input_data)

if __name__ == '__main__':
    main()
