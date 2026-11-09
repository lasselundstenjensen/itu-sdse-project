from vial_classifier.pipelines import should_promote


def test_first_model_is_always_promoted():
    assert should_promote(0.5, None) is True


def test_candidate_must_strictly_beat_production():
    assert should_promote(0.9, 0.8) is True
    assert should_promote(0.8, 0.8) is False
    assert should_promote(0.7, 0.8) is False


def test_missing_candidate_score_is_not_promoted():
    assert should_promote(None, 0.8) is False
