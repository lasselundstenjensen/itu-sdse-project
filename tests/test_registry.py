from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mlflow.exceptions import MlflowException

from vial_classifier import config
from vial_classifier.modeling import registry


class FakeVersion(dict):
    version = "2"
    run_id = "xyz"


def prod_run(score):
    return SimpleNamespace(data=SimpleNamespace(metrics={"test_f1_score": score}))


def test_get_production_model_uses_the_production_alias():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="3", run_id="abc")

    uri, version, run_id = registry.get_production_model(client)

    assert uri == f"models:/{config.MODEL_NAME}@Production"
    assert (version, run_id) == ("3", "abc")


def test_get_production_model_falls_back_to_the_production_stage():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("no alias")
    client.search_model_versions.return_value = [FakeVersion(current_stage="Production")]

    uri, version, run_id = registry.get_production_model(client)

    assert uri == f"models:/{config.MODEL_NAME}/2"
    assert (version, run_id) == ("2", "xyz")


def test_get_production_model_returns_none_when_nothing_is_in_production():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("no alias")
    client.search_model_versions.return_value = []

    assert registry.get_production_model(client) == (None, None, None)


def test_select_run_to_promote_picks_the_best_run_when_nothing_is_in_production():
    best = {"run_id": "new", "metrics.test_f1_score": 0.8}
    assert registry.select_run_to_promote(best, None) == "new"


def test_select_run_to_promote_only_promotes_a_strictly_better_model():
    best = {"run_id": "new", "metrics.test_f1_score": 0.8}
    with patch.object(registry.mlflow, "get_run", return_value=prod_run(0.7)):
        assert registry.select_run_to_promote(best, "prod") == "new"
    with patch.object(registry.mlflow, "get_run", return_value=prod_run(0.8)):
        assert registry.select_run_to_promote(best, "prod") is None
    with patch.object(registry.mlflow, "get_run", return_value=prod_run(0.9)):
        assert registry.select_run_to_promote(best, "prod") is None


def test_promote_model_moves_the_version_to_production_and_sets_the_alias():
    client = MagicMock()
    client.search_model_versions.return_value = [SimpleNamespace(run_id="r1", version="3")]
    client.get_model_version.side_effect = [
        {"current_stage": "Staging"},
        {"current_stage": "Production"},
    ]

    version = registry.promote_model(client, "r1")

    assert version == "3"
    client.transition_model_version_stage.assert_called_once_with(
        name=config.MODEL_NAME, version="3", stage="Production", archive_existing_versions=True
    )
    client.set_registered_model_alias.assert_called_once_with(
        name=config.MODEL_NAME, alias="Production", version="3"
    )
