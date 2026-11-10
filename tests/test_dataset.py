from pathlib import Path
from vial_classifier.dataset import split_dataset


def make_fake_dataset(n_clean=100, n_defective=400):
    clean = [(Path(f"clean_{i}.jpg"), 0) for i in range(n_clean)]
    defective = [(Path(f"defective_{i}.jpg"), 1) for i in range(n_defective)]
    return clean + defective


def test_split_is_stratified():
    for part in split_dataset(make_fake_dataset()):
        share_clean = sum(1 for _, label in part if label == 0) / len(part)
        assert 0.15 < share_clean < 0.25


def test_split_keeps_all_items_without_overlap():
    dataset = make_fake_dataset()
    train, val, test = split_dataset(dataset)
    assert len(train) + len(val) + len(test) == len(dataset)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)


def test_split_is_deterministic():
    assert split_dataset(make_fake_dataset()) == split_dataset(make_fake_dataset())