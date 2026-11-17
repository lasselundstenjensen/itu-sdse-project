from PIL import Image
from vial_classifier.features import define_transforms, GlassVialDataset

def test_val_transform_output_shape():
    _, val_tf = define_transforms()
    img = Image.new("RGB", (200, 100))
    assert val_tf(img).shape == (3, 64, 64)

def test_dataset_returns_image_and_label(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (50, 50)).save(path)
    ds = GlassVialDataset([path], [1])
    image, label = ds[0]
    assert len(ds) == 1
    assert label == 1