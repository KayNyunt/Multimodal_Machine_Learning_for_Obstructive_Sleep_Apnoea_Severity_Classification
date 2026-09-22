from pathlib import Path

# INPUT PATHS
CLINICAL_CSV = Path("/Users/OSA/NSRR_clinical_data/original_data/"
                    "homepap-baseline-harmonized-dataset-0.2.0.csv")

EDF_LAB_FULL = Path("/Users/OSA/EDFs/lab/full")
EDF_LAB_SPLIT = Path("/Users/OSA/EDFs/lab/split")
EDF_HOME = Path("/Users/OSA/EDFs/home")

ANNOTATIONS_LAB = Path("/Users/OSA/annotations-events-nsrr/lab/full")
ANNOTATIONS_HOME = Path("/Users/OSA/annotations-events-nsrr/home")


# OUTPUT PATHS
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

LAB_FEATURES_CSV = OUTPUT_DIR / "features_lab.csv"
HOME_FEATURES_CSV = OUTPUT_DIR / "features_home.csv"
OVERLAP_CSV = OUTPUT_DIR / "cohort_overlap.csv"

# CHANNEL NAME CANDIDATES
# HomePAP was recorded at seven centres, each using its own hardware,
# so the same signal appears under different names. These
# lists are tried in order by utils.resolve_channel().

SPO2_CANDIDATES = ["SAO2", "SpO2", "SPO2", "SA02", "SaO2", "Sat", "SAT"]
ECG_CANDIDATES = ["ECG1", "ECG", "EKG1", "EKG", "ECG-0", "ECGL"]
AIRFLOW_CANDIDATES = ["CannulaFlow", "CFLOW", "NPV", "Nasal",
                      "XFlow_PDS", "Airflow", "Thermistor"]
CHEST_CANDIDATES = ["Chest", "CHEST", "Thorax", "THOR"]
ABDOMEN_CANDIDATES = ["ABD", "Abdomen", "ABDO"]
PLETH_CANDIDATES = ["Pleth", "PLETH", "RD-Pleth"]

MODALITIES = {
    "spo2": SPO2_CANDIDATES,
    "ecg": ECG_CANDIDATES,
    "airflow": AIRFLOW_CANDIDATES,
    "chest": CHEST_CANDIDATES,
    "abdomen": ABDOMEN_CANDIDATES,
    "pleth": PLETH_CANDIDATES,
}

AHI_THRESHOLDS = [5, 15, 30]
SEVERITY_ORDER = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
LABEL_NAMES = {v: k for k, v in SEVERITY_ORDER.items()}

# CLINICAL FEATURES
CLINICAL_RENAME = {
    "nsrr_ahi_hp4u_aasm15": "AHI",
    "nsrr_sex": "Sex",
    "nsrr_age": "Age",
    "nsrr_race": "Race",
    "nsrr_ethnicity": "Ethnicity",
    "nsrr_bmi": "BMI",
    "nsrr_bp_systolic": "Systolic_bp",
    "nsrr_bp_diastolic": "Diastolic_bp",
    "nsrr_ttldursp_f1": "Total_Sleep_Duration",
}

CLINICAL_CATEGORICAL = ["Sex", "Race", "Ethnicity"]
CLINICAL_NUMERIC = ["Age", "BMI", "Systolic_bp", "Diastolic_bp"]
CLINICAL_FEATURES = CLINICAL_CATEGORICAL + CLINICAL_NUMERIC

# LEAKAGE CONTROL
LEAKAGE_COLS = [
    "AHI",
    "Total_Sleep_Duration",
    "total_sleep_hours",
    "ahi_from_events",
    "Severity",
    "nsrrid", "visit",
]

# QUALITY-CONTROL DIAGNOSTIC COLUMNS  --  excluded from modelling
QC_DIAGNOSTIC_COLS = ["valid_window_hours"]

# EXPERIMENT SETTINGS
RANDOM_STATE = 42
N_SPLITS = 5
PRIMARY_METRIC = "balanced_accuracy"
OVERLAP_STRATEGY = "exclude_from_lab"

WINDOW_OUTPUT_DIR = OUTPUT_DIR / "windows"
WINDOW_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)