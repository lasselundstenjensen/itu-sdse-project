from vial_classifier import config


def test_paths_are_inside_project():
    assert config.IMAGE_DIR.is_relative_to(config.PROJ_ROOT)
    assert config.ARTIFACT_DIR.is_relative_to(config.PROJ_ROOT)


def test_split_ratios_sum_to_one():
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-9