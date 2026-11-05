from vial_classifier import config


def test_paths_are_inside_project():
    for path in (config.DATA_DIR, config.MODELS_DIR, config.ARTIFACT_DIR, config.EXPORT_DIR):
        assert path.is_relative_to(config.PROJ_ROOT)


def test_dvc_pointers_exist_for_metadata():
    assert config.TRAIN_METADATA_PATH.with_suffix(".csv.dvc").exists()
    assert config.TEST_METADATA_PATH.with_suffix(".csv.dvc").exists()


def test_split_ratios_sum_to_one():
    assert abs(config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO - 1.0) < 1e-9


def test_binary_class_names():
    assert set(config.CLASS_NAMES) == {0, 1}
    assert config.NUM_CLASSES == len(config.CLASS_NAMES)
