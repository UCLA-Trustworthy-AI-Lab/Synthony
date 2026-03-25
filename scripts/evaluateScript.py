from pathlib import Path
from evaluator import EvaluationPipeline 
import pandas as pd  # only needed if EvaluationPipeline wants DataFrames

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

print("STARTING evaluateScript")
# Root dirs
real_root = Path("test_data")     # folder with real CSVs
synth_root = Path("synth_data")   # folder with subdirs per dataset
save_root = Path("results")       # evaluation outputs
save_root.mkdir(exist_ok=True)


column_name_to_datatype = {
 "News": {
        " timedelta": "numerical",
        " n_tokens_title": "numerical",
        " n_tokens_content": "numerical",
        " n_unique_tokens": "numerical",
        " n_non_stop_words": "numerical",
        " n_non_stop_unique_tokens": "numerical",
        " num_hrefs": "numerical",
        " num_self_hrefs": "numerical",
        " num_imgs": "numerical",
        " num_videos": "numerical",
        " average_token_length": "numerical",
        " num_keywords": "numerical",
        " data_channel_is_lifestyle": "numerical",
        " data_channel_is_entertainment": "numerical",
        " data_channel_is_bus": "numerical",
        " data_channel_is_socmed": "numerical",
        " data_channel_is_tech": "numerical",
        " data_channel_is_world": "numerical",
        " kw_min_min": "numerical",
        " kw_max_min": "numerical",
        " kw_avg_min": "numerical",
        " kw_min_max": "numerical",
        " kw_max_max": "numerical",
        " kw_avg_max": "numerical",
        " kw_min_avg": "numerical",
        " kw_max_avg": "numerical",
        " kw_avg_avg": "numerical",
        " self_reference_min_shares": "numerical",
        " self_reference_max_shares": "numerical",
        " self_reference_avg_sharess": "numerical",
        " weekday_is_monday": "numerical",
        " weekday_is_tuesday": "numerical",
        " weekday_is_wednesday": "numerical",
        " weekday_is_thursday": "numerical",
        " weekday_is_friday": "numerical",
        " weekday_is_saturday": "numerical",
        " weekday_is_sunday": "numerical",
        " is_weekend": "numerical",
        " LDA_00": "numerical",
        " LDA_01": "numerical",
        " LDA_02": "numerical",
        " LDA_03": "numerical",
        " LDA_04": "numerical",
        " global_subjectivity": "numerical",
        " global_sentiment_polarity": "numerical",
        " global_rate_positive_words": "numerical",
        " global_rate_negative_words": "numerical",
        " rate_positive_words": "numerical",
        " rate_negative_words": "numerical",
        " avg_positive_polarity": "numerical",
        " min_positive_polarity": "numerical",
        " max_positive_polarity": "numerical",
        " avg_negative_polarity": "numerical",
        " min_negative_polarity": "numerical",
        " max_negative_polarity": "numerical",
        " title_subjectivity": "numerical",
        " title_sentiment_polarity": "numerical",
        " abs_title_subjectivity": "numerical",
        " abs_title_sentiment_polarity": "numerical",
        " shares": "numerical",
    },
    "Bean": {
        "Area": "numerical",
        "Perimeter": "numerical",
        "MajorAxisLength": "numerical",
        "MinorAxisLength": "numerical",
        "AspectRation": "numerical",
        "Eccentricity": "numerical",
        "ConvexArea": "numerical",
        "EquivDiameter": "numerical",
        "Extent": "numerical",
        "Solidity": "numerical",
        "roundness": "numerical",
        "Compactness": "numerical",
        "ShapeFactor1": "numerical",
        "ShapeFactor2": "numerical",
        "ShapeFactor3": "numerical",
        "ShapeFactor4": "numerical",
        "Class": "categorical",
    },

    "Obesity": {
        "Gender": "categorical",
        "Age": "numerical",
        "Height": "numerical",
        "Weight": "numerical",
        "family_history_with_overweight": "categorical",
        "FAVC": "categorical",
        "FCVC": "numerical",
        "NCP": "numerical",
        "CAEC": "categorical",
        "SMOKE": "categorical",
        "CH2O": "numerical",
        "SCC": "categorical",
        "FAF": "numerical",
        "TUE": "numerical",
        "CALC": "categorical",
        "MTRANS": "categorical",
        "NObeyesdad": "categorical",   # target
    },

    "HTRU2": {
        "Var1": "numerical",
        "Var2": "numerical",
        "Var3": "numerical",
        "Var4": "numerical",
        "Var5": "numerical",
        "Var6": "numerical",
        "Var7": "numerical",
        "Var8": "numerical",
        "Class": "categorical",
    },

    "IndianLiverPatient": {
        "Age": "numerical",
        "Gender": "categorical",
        "Total_Bilirubin": "numerical",
        "Direct_Bilirubin": "numerical",
        "Alkaline_Phosphotase": "numerical",
        "Alamine_Aminotransferase": "numerical",
        "Aspartate_Aminotransferase": "numerical",
        "Total_Protiens": "numerical",
        "Albumin": "numerical",
        "Albumin_and_Globulin_Ratio": "numerical",
        "Dataset": "categorical",   # target
    },
  
    "wilt": {
        "GLCM_Pan": "numerical",
        "Mean_G": "numerical",
        "Mean_R": "numerical",
        "Mean_NIR": "numerical",
        "SD_Plan": "numerical",
        "class": "categorical",
    },

    "abalone": {
        "Sex": "categorical",
        "Length": "numerical",
        "Diameter": "numerical",
        "Height": "numerical",
        "Whole weight": "numerical",
        "Shucked weight": "numerical",
        "Viscera weight": "numerical",
        "Shell weight": "numerical",
        "Rings": "numerical",
    },

    "faults": {
        "X_Minimum": "numerical",
        "X_Maximum": "numerical",
        "Y_Minimum": "numerical",
        "Y_Maximum": "numerical",
        "Pixels_Areas": "numerical",
        "X_Perimeter": "numerical",
        "Y_Perimeter": "numerical",
        "Sum_of_Luminosity": "numerical",
        "Minimum_of_Luminosity": "numerical",
        "Maximum_of_Luminosity": "numerical",
        "Length_of_Conveyer": "numerical",
        "TypeOfSteel_A300": "numerical",
        "TypeOfSteel_A400": "numerical",
        "Steel_Plate_Thickness": "numerical",
        "Edges_Index": "numerical",
        "Empty_Index": "numerical",
        "Square_Index": "numerical",
        "Outside_X_Index": "numerical",
        "Edges_X_Index": "numerical",
        "Edges_Y_Index": "numerical",
        "Outside_Global_Index": "numerical",
        "LogOfAreas": "numerical",
        "Log_X_Index": "numerical",
        "Log_Y_Index": "numerical",
        "Orientation_Index": "numerical",
        "Luminosity_Index": "numerical",
        "SigmoidOfAreas": "numerical",
        "Pastry": "numerical",
        "Z_Scratch": "numerical",
        "K_Scatch": "numerical",
        "Stains": "numerical",
        "Dirtiness": "numerical",
        "Bumps": "numerical",
        "Other_Faults": "numerical",
    },

    "titanic": {
        "Survived": "numerical",
        "Pclass": "numerical",
        "Sex": "categorical",
        "Age": "numerical",
        "SibSp": "numerical",
        "Parch": "numerical",
        "Fare": "numerical",
        "Embarked": "categorical",
    },
}

configs_targets = {
	"News": "shares",
	"Bean": "Class",
        "faults": "pastry",
	"Obesity": "NObeyesdad",
	"HTRU2": "Class",
	"IndianLiverPatient": "Dataset",
	"wilt": "class",
	"abalone": "Rings",
	"titanic": "Embarked"

}

import pandas as pd

MISSING = "__MISSING__"

def resolve_target_col(real_cols, desired):
    # match case/space-insensitively to avoid metadata KeyErrors
    desired_norm = desired.strip().lower()
    norm_map = {c.strip().lower(): c for c in real_cols}
    if desired in real_cols:
        return desired
    if desired_norm in norm_map:
        return norm_map[desired_norm]
    raise KeyError(f"Target '{desired}' not found. Real columns: {list(real_cols)}")

def clean_and_align(real_df, synth_df, target_col):
    real = real_df.copy()
    synth = synth_df.copy()

    # normalize column names
    real.columns = [c.strip() for c in real.columns]
    synth.columns = [c.strip() for c in synth.columns]

    target_col = resolve_target_col(real.columns, target_col)

    # align columns (drop extras, error if missing)
    missing = [c for c in real.columns if c not in synth.columns]
    if missing:
        raise ValueError(f"Synth missing columns: {missing}")
    extra = [c for c in synth.columns if c not in real.columns]
    if extra:
        synth = synth.drop(columns=extra)

    synth = synth[real.columns]

    # drop rows with missing target (utility models + encoders hate NA in y)
    real = real.dropna(subset=[target_col]).copy()
    synth = synth.dropna(subset=[target_col]).copy()

    # decide which columns are numeric based on REAL data
    numeric_cols = set()
    medians = {}

    for c in real.columns:
        if c == target_col:
            continue
        num = pd.to_numeric(real[c], errors="coerce")
        # treat as numeric if it is mostly numeric
        if num.notna().mean() > 0.95:
            numeric_cols.add(c)
            med = num.median()
            if pd.isna(med):
                med = 0.0
            medians[c] = float(med)

    # now coerce BOTH real & synth consistently
    for c in real.columns:
        if c == target_col:
            # force plain python strings; avoid pandas "string" dtype (pd.NA)
            real[c] = real[c].astype(object).where(real[c].notna(), MISSING).astype(str)
            synth[c] = synth[c].astype(object).where(synth[c].notna(), MISSING).astype(str)
            continue

        if c in numeric_cols:
            med = medians[c]
            real[c] = pd.to_numeric(real[c], errors="coerce").astype("float64").fillna(med)
            synth[c] = pd.to_numeric(synth[c], errors="coerce").astype("float64").fillna(med)
        else:
            # categorical/text: force plain python strings + fill missing sentinel
            real[c] = real[c].astype(object).where(real[c].notna(), MISSING).astype(str)
            synth[c] = synth[c].astype(object).where(synth[c].notna(), MISSING).astype(str)

    return real, synth, target_col


# Loop over each real CSV
for real_path in sorted(real_root.glob("*.csv")):
    dataset_name = real_path.stem
    print(f"\n=== Dataset: {dataset_name} ===")
    
    if dataset_name == "News":
    	continue

    if dataset_name not in configs_targets:
        print(f"*** No target mapping for {dataset_name}, skipping")
        continue

    dataset_synth_dir = synth_root / dataset_name
    if not dataset_synth_dir.is_dir():
        print(f"*** No synthetic folder for {dataset_name} at {dataset_synth_dir}, skipping")
        continue

    real_df_raw = pd.read_csv(real_path)
    target_col_cfg = configs_targets[dataset_name]

    for synth_path in sorted(dataset_synth_dir.glob("*.csv")):
        synth_name = synth_path.stem
        print(f"--- Evaluating synthetic: {synth_name} ---")
        synth_df_raw = pd.read_csv(synth_path)

        real_df, synth_df, target_col = clean_and_align(real_df_raw, synth_df_raw, target_col_cfg)

        config = {
            "target_column": target_col,
            "target": target_col,
            "fidelity_metrics": ["SumStats", "ColumnShape"],
	    "privacy_metrics": ["DCR"],
            "holdout_size": 0.2,
            "holdout_seed": 0,

            # Optional: if your factory doesn't have Anonymeter registered, stop requesting it:
            # "privacy_metrics": ["DCR"],   # or []
        }

        run_save_path = save_root / dataset_name / synth_name
        run_save_path.mkdir(parents=True, exist_ok=True)

        evaluation_pipeline = EvaluationPipeline(
            real_data=real_df,
            synth_data=synth_df,
            column_name_to_datatype=column_name_to_datatype[dataset_name],
            config=config,
            save_path=run_save_path
        )

        evaluation_pipeline.run_pipeline()

