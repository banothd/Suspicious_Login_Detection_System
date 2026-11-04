import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path
from datetime import datetime


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Suspicious Login Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    BASE_DIR / "best_rf_model.joblib",
    BASE_DIR / "deployment" / "best_rf_model.joblib"
]

MODEL_PATH = next(
    (path for path in MODEL_PATHS if path.is_file()),
    None
)

DATA_PATH = BASE_DIR / "login_dataset.csv"
BACKGROUND_PATH = BASE_DIR / "background.png"


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

if MODEL_PATH is None:
    st.error(
        "Model file not found. Place best_rf_model.joblib "
        "beside app.py or inside the deployment folder."
    )
    st.stop()


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# =========================================================
# 3. LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset(path):
    """
    Load the CSV file using supported encodings.
    """

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            data = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )

            # Remove extra spaces from column names
            data.columns = data.columns.str.strip()

            return data

        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Unable to read the CSV file. "
        "Please check the file encoding."
    )


df = None

if DATA_PATH.is_file():
    try:
        df = load_dataset(str(DATA_PATH))

    except Exception as error:
        st.error("Failed to load the dataset.")
        st.error(str(error))

else:
    st.error(
        f"Dataset not found: {DATA_PATH}"
    )
# =========================================================
# 4. DROPDOWN HELPER
# =========================================================

def get_options(column, limit=200):

    if df is None or column not in df.columns:
        return ["Unknown"]

    values = (
        df[column]
        .dropna()
        .astype(str)
        .value_counts()
        .head(limit)
        .index
        .tolist()
    )

    options = sorted(set(values))

    if "Unknown" not in options:
        options.insert(0, "Unknown")

    return options or ["Unknown"]


# =========================================================
# 5. MODEL HELPER FUNCTIONS
# =========================================================

def get_classifier(estimator):

    if hasattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]

    return estimator


def get_positive_class_index(estimator):

    classifier = get_classifier(estimator)
    classes = getattr(classifier, "classes_", None)

    if classes is None:
        raise ValueError("Model class labels are unavailable.")

    positive_labels = {
        "1",
        "true",
        "yes",
        "account takeover",
        "account_takeover"
    }

    for index, label in enumerate(classes):

        if str(label).strip().lower() in positive_labels:
            return index

    raise ValueError(
        "Could not identify the account-takeover class. "
        f"Model classes: {list(classes)}"
    )


def align_input_columns(estimator, dataframe):

    expected_columns = getattr(
        estimator,
        "feature_names_in_",
        None
    )

    if expected_columns is None and hasattr(
        estimator, "named_steps"
    ):

        for step in estimator.named_steps.values():

            expected_columns = getattr(
                step,
                "feature_names_in_",
                None
            )

            if expected_columns is not None:
                break

    if expected_columns is None:
        return dataframe

    expected_columns = list(expected_columns)

    missing_columns = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The model requires additional features: "
            + ", ".join(missing_columns)
        )

    return dataframe[expected_columns]


def get_risk_level(probability):

    # Demonstration thresholds only.
    # Validate these using model validation data.

    if probability >= 0.70:
        return "High"

    elif probability >= 0.30:
        return "Medium"

    return "Low"


def location_input(label, column, limit, default="Unknown"):

    """Allow dropdown selection or manual text entry."""

    mode = st.radio(
        f"{label} input method",
        ["Choose from list", "Enter manually"],
        horizontal=True,
        key=f"{column}_mode"
    )

    if mode == "Choose from list":

        options = get_options(column, limit)

        return st.selectbox(
            label,
            options,
            key=f"{column}_select"
        )

    return st.text_input(
        label,
        value=default,
        key=f"{column}_manual",
        help="Enter the value you want to test."
    )


def feature_input(label, column, limit, default="Unknown"):

    """Allow dropdown selection or manual text entry."""

    mode = st.radio(
        f"{label} input method",
        ["Choose from list", "Enter manually"],
        horizontal=True,
        key=f"{column}_mode"
    )

    if mode == "Choose from list":

        return st.selectbox(
            label,
            get_options(column, limit),
            key=f"{column}_select"
        )

    return st.text_input(
        label,
        value=default,
        key=f"{column}_manual",
        help="Enter a value to test with the model."
    )


# =========================================================
# 6. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# 7. FULL-PAGE BACKGROUND IMAGE
# =========================================================

if BACKGROUND_PATH.is_file():

    with open(BACKGROUND_PATH, "rb") as image_file:

        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    background_css = f"""
    <style>

    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(
                rgba(5, 15, 30, 0.78),
                rgba(5, 15, 30, 0.84)
            ),
            url("data:image/jpeg;base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(7, 20, 38, 0.90);
    }}

    h1, h2, h3, p, label {{
        color: #F1F5F9;
    }}

    </style>
    """

    st.markdown(
        background_css,
        unsafe_allow_html=True
    )


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Login Security")

    st.write(
        "Assess login activity and review possible "
        "account-takeover risks."
    )

    st.divider()

    st.subheader("Session Controls")

    if st.button(
        "Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.history = []
        st.session_state.last_prediction = None

        st.rerun()

    st.divider()

    st.caption("Model: Random Forest")
    st.caption("Application: Interactive demonstration")


# =========================================================
# 9. APPLICATION HEADER
# =========================================================

st.title("Suspicious Login Detection")

st.write(
    """
    Enter the login information below to assess whether
    the activity may involve an account takeover.
    """
)

st.info(
    "The result is a machine-learning prediction, not proof "
    "of fraud. Review suspicious activity before taking action."
)

# =========================================================
# 9. LOGIN INPUT FORM
# =========================================================

with st.form("login_form"):

    # -----------------------------------------------------
    # 1. LOGIN LOCATION
    # -----------------------------------------------------

    st.subheader("1. Login Location")
    st.caption(
        "Enter the location from which the login was attempted."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        country_options = get_options("Country", 60)

        country = st.selectbox(
            "Country",
            country_options,
            index=0,
            help="Select the country associated with the login."
        )

    with col2:
        region_options = get_options("Region", 100)

        region = st.selectbox(
            "Region / State",
            region_options,
            index=0,
            help="Select the region or state."
        )

    with col3:
        city_options = get_options("City", 150)

        city = st.selectbox(
            "City",
            city_options,
            index=0,
            help="Select the city."
        )

    st.divider()

    # -----------------------------------------------------
    # 2. DEVICE AND BROWSER
    # -----------------------------------------------------

    st.subheader("2. Device and Browser")
    st.caption(
        "Select the browser and device used for the login."
    )

    col4, col5 = st.columns(2)

    with col4:
        browser_options = get_options(
            "Browser Name and Version", 100
        )

        browser = st.selectbox(
            "Browser",
            browser_options,
            index=0,
            help="Choose the browser used during login."
        )

        os_options = get_options(
            "OS Name and Version", 100
        )

        os_name = st.selectbox(
            "Operating System",
            os_options,
            index=0,
            help="Choose the operating system."
        )

    with col5:
        device_options = get_options("Device Type", 20)

        device_type = st.selectbox(
            "Device Type",
            device_options,
            index=0,
            help="Select Desktop, Mobile, or another available device type."
        )

        user_agent_options = get_options(
            "User Agent String", 100
        )

        user_agent = st.selectbox(
            "Browser / Device Details",
            user_agent_options,
            index=0,
            help="Select the user-agent string associated with the login."
        )

    st.divider()

    # -----------------------------------------------------
    # 3. LOGIN ACTIVITY
    # -----------------------------------------------------

    st.subheader("3. Login Activity")
    st.caption(
        "Enter details about the login attempt and network."
    )

    col6, col7, col8 = st.columns(3)

    with col6:
        login_successful = st.selectbox(
            "Was the Login Successful?",
            [True, False],
            format_func=lambda value: "Yes" if value else "No",
            help="Select whether the login succeeded."
        )

    with col7:
        login_hour = st.selectbox(
            "Login Time",
            list(range(24)),
            index=12,
            format_func=lambda hour: f"{hour:02d}:00",
            help="Select the hour when the login occurred."
        )

    with col8:
        rtt = st.number_input(
            "Network Response Time (ms)",
            min_value=0.0,
            max_value=60000.0,
            value=100.0,
            step=10.0,
            help="Enter the network round-trip time in milliseconds."
        )

    # -----------------------------------------------------
    # 4. NETWORK DETAILS
    # -----------------------------------------------------

    st.markdown("**Network Details**")

    asn_options = get_options("ASN", 100)

    asn = st.selectbox(
        "Network (ASN)",
        asn_options,
        index=0,
        help="Select the Autonomous System Number associated with the login."
    )

    st.divider()

    # -----------------------------------------------------
    # SUBMIT
    # -----------------------------------------------------

    submitted = st.form_submit_button(
        "Check Login",
        use_container_width=True
    )
# =========================================================
# 11. PREDICTION
# =========================================================

if submitted:

    input_values = {
        "Login Timestamp": pd.Timestamp.now(),

        "Round-Trip Time [ms]": rtt,

        "Country": country.strip() or "Unknown",

        "Region": region.strip() or "Unknown",

        "City": city.strip() or "Unknown",

        "ASN": str(asn).strip() or "Unknown",

        "User Agent String": user_agent.strip() or "Unknown",

        "Browser Name and Version":
            browser.strip() or "Unknown",

        "OS Name and Version":
            os_name.strip() or "Unknown",

        "Device Type": device_type.strip() or "Unknown",

        "Login Successful": login_successful,

        "Login Hour": login_hour
    }

    try:

        input_data = pd.DataFrame([input_values])

        input_data = align_input_columns(
            model,
            input_data
        )

        # Predict the account-takeover class.

        prediction = model.predict(input_data)[0]

        positive_labels = {
            "1",
            "true",
            "yes",
            "account takeover",
            "account_takeover"
        }

        prediction_label = str(
            prediction
        ).strip().lower()

        classifier = get_classifier(model)

        classes = getattr(
            classifier,
            "classes_",
            None
        )

        if classes is not None:

            if not any(
                str(label).strip().lower() in positive_labels
                for label in classes
            ):

                raise ValueError(
                    "Cannot identify the account-takeover class. "
                    f"Model classes: {list(classes)}"
                )

        is_takeover = (
            prediction_label in positive_labels
        )

        # Calculate the probability of account takeover.

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            positive_index = get_positive_class_index(
                model
            )

            probability = float(
                probabilities[positive_index]
            )

        # Determine risk level.

        risk_level = None

        if probability is not None:

            risk_level = get_risk_level(
                probability
            )

        # Save the latest result.

        st.session_state.last_prediction = {
            "is_takeover": is_takeover,
            "probability": probability,
            "risk_level": risk_level,
            "input_values": input_values,
            "input_data": input_data,
            "prediction_time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

        # Save prediction history.

        st.session_state.history.append({

            "Checked At": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Country": country,

            "Region": region,

            "City": city,

            "Device": device_type,

            "Browser": browser,

            "Login Successful": (
                "Yes" if login_successful else "No"
            ),

            "Prediction": (
                "Account Takeover: YES"
                if is_takeover
                else "Account Takeover: NO"
            ),

            "Estimated Probability (%)": (
                round(probability * 100, 4)
                if probability is not None
                else "Unavailable"
            ),

            "Risk Level": (
                risk_level
                if risk_level is not None
                else "Unavailable"
            )
        })

    except Exception as error:

        st.error(
            "Prediction failed. Check that the input features "
            "and categories match the trained model."
        )

        st.exception(error)


# =========================================================
# 12. DISPLAY PREDICTION OUTPUT
# =========================================================

result = st.session_state.last_prediction

if result is not None:

    st.divider()

    st.header("Account Security Assessment")

    probability = result["probability"]

    risk_level = result["risk_level"]

    is_takeover = result["is_takeover"]

    # -----------------------------------------------------
    # MAIN PREDICTION
    # -----------------------------------------------------

    if is_takeover:

        st.error(
            "ACCOUNT TAKEOVER: YES — "
            "Possible Account Takeover"
        )

    else:

        st.success(
            "ACCOUNT TAKEOVER: NO — "
            "Account Takeover Not Predicted"
        )

    # -----------------------------------------------------
    # RISK ASSESSMENT
    # -----------------------------------------------------

    st.subheader("Risk Assessment")

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Estimated Takeover Probability",
            (
                f"{probability:.2%}"
                if probability is not None
                else "Unavailable"
            )
        )

    with metric2:

        st.metric(
            "Risk Level",
            risk_level if risk_level else "Unavailable"
        )

    if probability is not None:

        st.progress(
            min(max(probability, 0.0), 1.0),
            text="Estimated probability of account takeover"
        )

    st.caption(
        "Probability is the model's estimate, not a guarantee "
        "that an account has or has not been compromised."
    )

    # -----------------------------------------------------
    # SECURITY RECOMMENDATION
    # -----------------------------------------------------

    st.subheader("Security Recommendation")

    if is_takeover:

        st.warning(
            """
            **Recommended action: Review this login.**

            - Verify the user's identity.
            - Review the login activity.
            - Follow your account-security procedure.
            """
        )

    elif risk_level == "High":

        st.warning(
            """
            The model predicted NO, but the estimated probability
            falls in the configured high-risk category.

            Review the login and consider additional verification.
            """
        )

    elif risk_level == "Medium":

        st.info(
            """
            **Recommended action: Review the login details.**

            Consider additional verification according to your
            organization's security procedures.
            """
        )

    else:

        st.info(
            """
            **Recommended action: Continue normal security checks.**

            A low estimated risk does not guarantee that a login
            is safe.
            """
        )

    # -----------------------------------------------------
    # LOGIN SUMMARY
    # -----------------------------------------------------

    st.subheader("Login Summary")

    values = result["input_values"]

    summary = pd.DataFrame([
        {
            "Detail": "Login Status",
            "Value": (
                "Successful"
                if values["Login Successful"]
                else "Failed"
            )
        },
        {
            "Detail": "Country",
            "Value": values["Country"]
        },
        {
            "Detail": "Region / State",
            "Value": values["Region"]
        },
        {
            "Detail": "City",
            "Value": values["City"]
        },
        {
            "Detail": "Device",
            "Value": values["Device Type"]
        },
        {
            "Detail": "Operating System",
            "Value": values["OS Name and Version"]
        },
        {
            "Detail": "Browser",
            "Value": values["Browser Name and Version"]
        },
        {
            "Detail": "Login Hour",
            "Value": f'{values["Login Hour"]:02d}:00'
        },
        {
            "Detail": "Network Response Time",
            "Value": (
                f'{values["Round-Trip Time [ms]"]:.0f} ms'
            )
        },
        {
            "Detail": "Network (ASN)",
            "Value": values["ASN"]
        }
    ])

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View model input details"):
        st.dataframe(
            result["input_data"],
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# 13. RECENT PREDICTION HISTORY
# =========================================================

st.divider()

st.header("Recent Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction History (CSV)",
        data=csv_data,
        file_name="login_prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No predictions yet. Submit a login assessment "
        "to display prediction history."
    )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "Suspicious Login Detection | Machine Learning Project | "
    "Predictions should be reviewed alongside other security evidence."
)