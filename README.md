This is the code that follows SCM8053_Dissertation 2025/2026.

The aim of this project is to classify obstructive sleep apnoea (OSA) severity into four classes (Normal / Mild / Moderate / Severe, by AHI) from the HomePAP dataset (National Sleep Research Resource).

The actual datasets are not uploaded into this repository.

config.py - The file contains every input and output path as well as every constant needed for the project.

utils.py - The file contains shared function used throughout the pipeline.

step_01_channel_names_checking_all_files.py - Confirms every modality's channel can actually be located in every recording.

step_02_cohort_overlap.py - Identifies participants who appear in both cohorts.

step_03_extract_features.py - Extracts 38 features across five physiological modalities 
                              plus a pulse-transit-time measure, from raw signal only.

step_04_feature_associations.py - Spearman correlation between every feature and severity, computed separately per cohort.

step_05_modelling.py - Ten algorithms of three configuration from both cohorts and a cross-cohort validation of Lab to Home.

step_06_tuning.py - Hyperparameter tuning (randomised search, five algorithms) and recursive feature elimination.

step_06b_SHAP_selection.py - An independent, SHAP-based feature selection.

step_07_validation_controls.py - To check how much is attributable to the smaller training set versus the change in recording equipment,
                                 plus a bidirectional transfer check (train on home, test on laboratory, as a symmetry test).

step_08_figures.py - Generate all figures used throughout the project.

step_09_ROC_Curves.py - One-vs-rest ROC and precision-recall curves for the best model in each cohort.

step_10_SHAP.py - SHAP feature importance and beeswarm plots for the best tree-based model.

step_11_Stacking.py - Stacking and voting ensembles built from the top-performing models, compared against the single best model.
