import pandas as pd
from src.data import load_dataset
from src.train import build_preprocessor, evaluate

def test_load_dataset_converts_uci_num_to_binary_target(tmp_path):
    source = tmp_path / "heart.csv"
    pd.DataFrame({"age": [45, 55], "num": [0, 2]}).to_csv(source, index=False)
    assert load_dataset(source)["target"].tolist() == [0, 1]

def test_preprocessor_handles_missing_value():
    frame = pd.DataFrame({"age": [45, None, 60], "chol": [210, 240, 200]})
    assert build_preprocessor(["age", "chol"]).fit_transform(frame).shape == (3, 2)

def test_evaluate_returns_requested_metrics():
    class FixedModel:
        def predict(self, frame): return [0, 1, 1, 0]
        def predict_proba(self, frame): return [[.8,.2],[.1,.9],[.2,.8],[.9,.1]]
    result = evaluate(FixedModel(), pd.DataFrame({"x":[1,2,3,4]}), pd.Series([0,1,1,0]))
    assert result == {"accuracy":1.0, "precision":1.0, "recall":1.0, "roc_auc":1.0}
