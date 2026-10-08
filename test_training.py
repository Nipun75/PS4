from train_model import train_model


def test_training_pipeline():
    search, metrics, feature_names = train_model(save=False)

    assert len(feature_names) == 30
    assert metrics["train_samples"] + metrics["test_samples"] == 569
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_precision"] <= 1.0
    assert 0.0 <= metrics["test_recall"] <= 1.0
    assert 0.0 <= metrics["test_f1"] <= 1.0
    assert search.best_params_["tree__criterion"] in {"gini", "entropy"}
