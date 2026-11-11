"""Train the two response models on the training split and save them.

Usage:  python src/train.py [--data your_experiment.csv] [--out model_artifacts]
Your CSV needs the feature columns in growth_model.FEATURES plus treatment_group (Control/Treatment) and converted (0/1).
Artifacts go to model_artifacts/ (git-ignored: joblib files depend on the scikit-learn version,
so they are rebuilt from data rather than committed).
"""
import argparse, json
import joblib, sklearn
from growth_model import ROOT, FEATURES, load_experiment, split_arms, fit_two_models

ART = ROOT / "model_artifacts"


def main():
    global ART
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out"); a = ap.parse_args()
    df = load_experiment(a.data) if a.data else load_experiment()
    missing = [c for c in FEATURES + ["treatment_group", "converted"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if a.out:
        from pathlib import Path; ART = Path(a.out)
    parts = split_arms(df)
    pre, ctrl, trt = fit_two_models(parts["Control"][0], parts["Treatment"][0])
    ART.mkdir(exist_ok=True)
    joblib.dump(pre, ART / "preprocessor.joblib"); joblib.dump(ctrl, ART / "control_model.joblib"); joblib.dump(trt, ART / "treatment_model.joblib")
    json.dump(dict(sklearn=sklearn.__version__, trained_on="75% training split of each arm",
                   n_control_train=len(parts["Control"][0]), n_treatment_train=len(parts["Treatment"][0])),
              open(ART / "metadata.json", "w"), indent=2)
    print(f"Saved models to {ART} (trained on {len(parts['Control'][0])} control + {len(parts['Treatment'][0])} treatment customers)")


if __name__ == "__main__":
    main()
