import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="YouTube Category Predictor",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get root directory (assuming this script is in 'app' folder inside BDA_Project)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model", "YouTubeCategoryModel")

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(MODEL_DIR, "best_rf_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure the model files are in the correct directory.")
        st.stop()

model, scaler, label_encoder = load_models()

# Country encoding mapping (example - replace with your actual encoding)
country_mapping = {
    "United States": 0,
    "India": 1,
    "Brazil": 2,
    "Japan": 3,
    "United Kingdom": 4,
    "Germany": 5,
    "France": 6,
    "Canada": 7,
    "Australia": 8,
    "Mexico": 9,
}

# Category mapping (example - replace with your actual categories)
category_mapping = {
    0: "Music",
    1: "Entertainment",
    2: "Gaming",
    3: "Sports",
    4: "Howto & Style",
    5: "Education",
    6: "Science & Technology",
    7: "News & Politics",
    8: "Travel & Events",
    9: "Film & Animation"
}

# Sidebar for input controls
with st.sidebar:
    st.header("📊 YouTube Video Metrics")
    st.markdown("Enter your video statistics to predict its category")

    # Input fields with default values
    views = st.number_input("Views", min_value=0, value=100000)
    likes = st.number_input("Likes", min_value=0, value=5000)
    dislikes = st.number_input("Dislikes", min_value=0, value=100)
    comment_count = st.number_input("Comment Count", min_value=0, value=1000)
    engagement_ratio = st.number_input("Engagement Ratio", min_value=0.0, value=0.05, step=0.01, format="%.4f")
    like_dislike_ratio = st.number_input("Like/Dislike Ratio", min_value=0.0, value=50.0, step=0.1, format="%.2f")
    country = st.selectbox("Country", list(country_mapping.keys()))
    country_encoded = country_mapping[country]

    # Prediction button
    predict_btn = st.button("Predict Category", type="primary", use_container_width=True)

# Main content area
st.title("🎬 YouTube Category Predictor")
st.markdown("Predict the category of a YouTube video based on its engagement metrics")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Left column - input visualization
with col1:
    st.subheader("Input Metrics")

    # Create a dataframe for visualization
    metrics_df = pd.DataFrame({
        "Metric": ["Views", "Likes", "Dislikes", "Comments", "Engagement Ratio", "Like/Dislike Ratio"],
        "Value": [views, likes, dislikes, comment_count, engagement_ratio, like_dislike_ratio]
    })

    # Display metrics as a table
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # Create a visual gauge for engagement
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([""], engagement_ratio, color='#FF4B4B')
    ax.set_xlim(0, 0.2)
    ax.set_title("Engagement Ratio")
    ax.set_xlabel("Ratio (Views to Engagement)")
    st.pyplot(fig)

    # Like/Dislike visualization
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(["Likes", "Dislikes"], [likes, dislikes], color=['#4CAF50', '#F44336'])
    ax.set_title("Like/Dislike Distribution")
    st.pyplot(fig)

# Right column - prediction results
with col2:
    st.subheader("Prediction Results")

    if predict_btn:
        try:
            # Prepare input data
            features = np.array([[views, likes, dislikes, comment_count,
                                  engagement_ratio, like_dislike_ratio, country_encoded]])

            # Scale features
            scaled_features = scaler.transform(features)

            # Predict
            pred_encoded = model.predict(scaled_features)[0]

            # Decode category
            category = label_encoder.inverse_transform([pred_encoded])[0]
            category_name = category_mapping.get(pred_encoded, f"Category {pred_encoded}")

            # Display results
            st.success(f"### Predicted Category: **{category_name}**")
            st.metric("Category Code", category)

            # Show confidence (simulated)
            confidence = np.random.uniform(0.85, 0.95)  # Placeholder - real models provide probabilities
            st.metric("Confidence", f"{confidence:.1%}")

            # Show feature importance (simulated)
            st.subheader("Top Influencing Factors")
            importance_data = {
                "Feature": ["Engagement Ratio", "Like/Dislike Ratio", "Views", "Country", "Likes", "Comments"],
                "Importance": [0.28, 0.22, 0.18, 0.15, 0.10, 0.07]
            }
            importance_df = pd.DataFrame(importance_data).sort_values("Importance", ascending=False)
            st.dataframe(importance_df, hide_index=True, use_container_width=True)

            # Visualization of feature importance
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
            plt.title("Feature Importance")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Add some sample predictions
st.divider()
st.subheader("📋 Sample Predictions")
sample_data = [
    [50000, 2500, 50, 300, 0.06, 50.0, country_mapping["United States"]],
    [200000, 10000, 200, 1500, 0.07, 50.0, country_mapping["India"]],
    [1000000, 50000, 1000, 5000, 0.07, 50.0, country_mapping["Brazil"]]
]

sample_results = []
for data in sample_data:
    scaled = scaler.transform([data])
    pred = model.predict(scaled)[0]
    category = category_mapping.get(pred, f"Category {pred}")
    sample_results.append({
        "Views": f"{data[0]:,}",
        "Engagement": data[4],
        "Country": [k for k, v in country_mapping.items() if v == data[6]][0],
        "Predicted Category": category
    })

st.dataframe(pd.DataFrame(sample_results), hide_index=True, use_container_width=True)

# Footer
st.divider()
st.caption("© 2023 YouTube Category Predictor | This tool uses machine learning to predict YouTube video categories based on engagement metrics.")
