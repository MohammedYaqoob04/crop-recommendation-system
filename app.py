import streamlit as st  # web app framework
import pandas as pd  # for building the input DataFrame the model expects
import joblib  # for loading the saved trained model


@st.cache_resource  # cache so the model loads ONCE per session, not on every button click
def load_model():
    # load the trained Random Forest model saved by train_model.py
    return joblib.load("model/crop_model.pkl")


# try to load the model; show a clear error instead of a raw traceback if it's missing
try:
    model = load_model()  # cached load
except FileNotFoundError:
    st.error("Model file not found at model/crop_model.pkl. Run train_model.py first.")  # tell the user exactly what to do
    st.stop()  # halt execution here so nothing below runs against a missing model


st.title("🌱 Crop Recommendation System")  # page title
st.write("Enter the agricultural conditions to get a suitable crop recommendation.")  # short instruction


# feature importance is GLOBAL to the trained model, not specific to any one prediction,
# so it's shown once, outside the form, with a caption that says exactly that
with st.expander("How the model decides (overall feature importance)"):
    # names must match the order used when the model was trained
    feature_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]  # display labels

    # model.feature_importances_ is the SAME for every prediction — it reflects the whole trained forest
    feature_importance = pd.Series(
        model.feature_importances_,  # importance scores from the trained model
        index=feature_names  # label each score
    ).sort_values(ascending=False)  # highest importance first

    st.caption("Overall importance across all crops — not specific to any single prediction below.")  # honest framing
    st.bar_chart(feature_importance)  # render as a bar chart


with st.form("crop_form"):  # group inputs so the app only reruns on submit, not on every keystroke

    st.subheader("Agricultural Parameters")  # section header

    N = st.number_input(  # nitrogen input, bounded to the dataset's real observed range
        "Nitrogen (N)",
        min_value=0.0,
        max_value=140.0,
        value=50.0
    )

    P = st.number_input(  # phosphorus input
        "Phosphorus (P)",
        min_value=5.0,
        max_value=145.0,
        value=50.0
    )

    K = st.number_input(  # potassium input
        "Potassium (K)",
        min_value=5.0,
        max_value=205.0,
        value=50.0
    )

    temperature = st.number_input(  # temperature input in Celsius
        "Temperature (°C)",
        min_value=8.83,
        max_value=43.68,
        value=25.0
    )

    humidity = st.number_input(  # relative humidity input
        "Humidity (%)",
        min_value=14.26,
        max_value=99.98,
        value=70.0
    )

    ph = st.number_input(  # soil pH input, valid range is 0-14 but bounded tighter to observed data
        "Soil pH",
        min_value=3.50,
        max_value=9.94,
        value=6.5
    )

    rainfall = st.number_input(  # rainfall input in mm
        "Rainfall (mm)",
        min_value=20.21,
        max_value=298.56,
        value=100.0
    )

    submitted = st.form_submit_button("Recommend Crop")  # single submit button for the whole form


if submitted:  # only run prediction logic after the user clicks submit

    # re-check every value against its valid range explicitly, instead of trusting
    # Streamlit's widget-level display, which can show an error while still holding
    # a stale valid value underneath — this makes validation explicit and visible
    validation_errors = []  # collect every problem instead of stopping at the first one

    if not (0.0 <= N <= 140.0):  # nitrogen bounds
        validation_errors.append(f"Nitrogen (N) = {N} is outside 0–140.")
    if not (5.0 <= P <= 145.0):  # phosphorus bounds
        validation_errors.append(f"Phosphorus (P) = {P} is outside 5–145.")
    if not (5.0 <= K <= 205.0):  # potassium bounds
        validation_errors.append(f"Potassium (K) = {K} is outside 5–205.")
    if not (8.83 <= temperature <= 43.68):  # temperature bounds
        validation_errors.append(f"Temperature = {temperature} is outside 8.83–43.68.")
    if not (14.26 <= humidity <= 99.98):  # humidity bounds
        validation_errors.append(f"Humidity = {humidity} is outside 14.26–99.98.")
    if not (3.50 <= ph <= 9.94):  # soil pH bounds
        validation_errors.append(f"Soil pH = {ph} is outside 3.50–9.94.")
    if not (20.21 <= rainfall <= 298.56):  # rainfall bounds
        validation_errors.append(f"Rainfall = {rainfall} is outside 20.21–298.56.")

    if validation_errors:  # if anything failed, show every failure and stop before predicting
        for error_message in validation_errors:  # list each problem clearly
            st.error(error_message)
        st.stop()  # halt so the model never runs on values you haven't confirmed are valid

    # column names here MUST exactly match the columns train_model.py trained on,
    # including case — check train_model.py uses "ph" (lowercase), not "pH"
    input_data = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })

    probabilities = model.predict_proba(input_data)[0]  # get probability for every crop class

    crop_names = model.classes_  # the crop labels in the same order as the probabilities

    crop_probabilities = pd.Series(
        probabilities,  # the probability values
        index=crop_names  # labeled by crop name
    ).sort_values(ascending=False)  # highest probability first

    st.success(
        f"Recommended Crop: {crop_probabilities.index[0].title()}"  # top prediction, title-cased for display
    )

    st.subheader("Top 3 Crop Recommendations")  # section header for the ranked list

    for crop, probability in crop_probabilities.head(3).items():  # loop over the top 3 only
        st.write(f"**{crop.title()}** — {probability:.1%}")  # show crop name and confidence as a percentage