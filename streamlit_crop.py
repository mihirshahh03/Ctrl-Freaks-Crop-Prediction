import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import chardet

# Function to load and preprocess data
def load_and_preprocess_data():
    with open(r"C:\Users\ronit\OneDrive\Desktop\College\Python\Hacker\generated_data.csv", 'rb') as f:
        result = chardet.detect(f.read())

    data = pd.read_csv(r"C:\Users\ronit\OneDrive\Desktop\College\Python\Hacker\generated_data.csv", encoding=result['encoding'])

    X = data.drop(columns=['Crop Recommended'])
    y = data['Crop Recommended']

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to train the Random Forest classifier
def train_random_forest(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

# Function to make predictions
def make_predictions(rf_classifier, farm_data):
    farm_data_reindexed = farm_data.reindex(columns=X_train.columns, fill_value=0)
    recommended_crop = rf_classifier.predict(farm_data_reindexed)
    return recommended_crop[0]

# Load and preprocess data
X_train, _, y_train, _ = load_and_preprocess_data()

# Train Random Forest classifier
rf_classifier = train_random_forest(X_train, y_train)

# Streamlit App
st.title("Precision Farming")

# Navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Crop Prediction"])

if page == "Home":
    # Home Page
    # Precision Farming Section
    st.header("Precision Farming")

    # Add content related to precision farming
    st.write("""
    Precision farming involves using technology to optimize agricultural practices and increase efficiency.
    """)

    # Vision Section
    st.header("Our Vision And The Problem")

    # Add content related to vision in precision farming
    st.write("""
             Precision Agriculture as a Lifeline:
"Precision agriculture emerged as our chosen path, a beacon of change amidst adversity. By leveraging technology, data, and innovation, we believe we can bring about a revolution in farming practices, heralding a new era of prosperity."
    As a team deeply moved by the plight of our farmers, we couldn't turn a blind eye to the challenges they face. The heart-wrenching tales of hardships, debt, and despair spurred us to seek a solution. We envisioned a future where technology becomes a beacon of hope for farmers, transforming their struggles into success stories.
             Our vision is not just about embracing technology; it's about embracing the lives of those who till the land. We understand that the resilience of our farmers forms the backbone of our nation, and it's our collective responsibility to ensure their well-being.

             
Inefficient resource utilization (water, fertilizers, pesticides).
Unpredictable crop yields due to weather variations and pests.
Environmental degradation, including soil erosion and water pollution.
Rising costs of agricultural inputs affecting profitability.
Overwhelming amounts of data without effective interpretation.
Market volatility and fluctuating crop prices.
Limited water availability and the need for efficient water management.
Adapting to the impact of climate change on traditional farming practices.
             Addressing the challenge of optimizing resource utilization, minimizing environmental impact, and enhancing crop yield through the implementation of DataScience techniques unto precision agriculture applications
    """)

    # Video Section
    st.header("Video: Precision Farming in Action")

    # Provide the path to your video file
    video_url = r"C:\Users\ronit\OneDrive\Desktop\College\Python\Hacker\video.mp4"
    st.video(video_url, format="video/mp4")

elif page == "Crop Prediction":
    # Machine Learning Page
    # Machine Learning Section
    st.header("Crop Recommendation")

    # Farm ID input
    farm_id_to_predict = st.number_input("Enter Farm ID to predict crop for:", value=1, step=1)

    # Load farm data for prediction
    farm_data = pd.DataFrame({"Farm ID": [farm_id_to_predict]})
    farm_data_reindexed = farm_data.reindex(columns=X_train.columns, fill_value=0)

    # Predict crop
    if st.button("Predict Crop"):
        recommended_crop = make_predictions(rf_classifier, farm_data_reindexed)
        st.success(f"Recommended crop for Farm ID {farm_id_to_predict}: {recommended_crop}")

# ABOUT Section
st.markdown("""
<style>
.about-section {
    background-color: black;
    padding: 30px;
    border-radius: 15px;
    color: white;
    width: 80%; /* Adjust width as needed */
    margin: auto; /* Center the about section */
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='about-section'>", unsafe_allow_html=True)
st.markdown("### ABOUT")
st.write("""
This Streamlit app showcases information about Precision Farming, the role of vision technology in agriculture,
and a machine learning model for crop recommendation.
For more details, contact us at [your_email@example.com](mailto:your_email@example.com).
""")
st.markdown("</div>", unsafe_allow_html=True)
