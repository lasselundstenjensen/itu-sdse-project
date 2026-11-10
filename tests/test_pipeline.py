from vial_classifier import pipeline


def test_data_pulls_data(monkeypatch):
    calls = []

    def fake_pull():
        calls.append("pull")

    monkeypatch.setattr(pipeline, "pull_data_from_dvc", fake_pull)

    pipeline.data()

    assert calls == ["pull"]