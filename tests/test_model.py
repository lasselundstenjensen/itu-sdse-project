import torch
from torch import nn

from modeling.model import SimpleCNN
from modeling.train import load_checkpoint, save_checkpoint


class NotebookSimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        inputs = self.pool(self.relu(self.conv1(inputs)))
        inputs = self.pool(self.relu(self.conv2(inputs)))
        inputs = inputs.view(-1, 64 * 16 * 16)
        inputs = self.relu(self.fc1(inputs))
        inputs = self.dropout(inputs)
        return self.fc2(inputs)


def test_output_shape_for_multiple_batch_sizes():
    model = SimpleCNN().eval()

    for batch_size in (1, 3, 8):
        with torch.no_grad():
            output = model(torch.randn(batch_size, 3, 64, 64))
        assert output.shape == (batch_size, 1)


def test_model_matches_notebook_with_same_seed_and_input():
    torch.manual_seed(42)
    reference_model = NotebookSimpleCNN().eval()
    torch.manual_seed(42)
    refactored_model = SimpleCNN().eval()
    torch.manual_seed(42)
    inputs = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        reference_output = reference_model(inputs)
        refactored_output = refactored_model(inputs)

    torch.testing.assert_close(refactored_output, reference_output, rtol=1e-6, atol=1e-7)


def test_saved_checkpoint_reloads_same_output(tmp_path):
    torch.manual_seed(42)
    original_model = SimpleCNN().eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        original_output = original_model(inputs)

    checkpoint_path = tmp_path / "vial_cnn_model.pth"
    save_checkpoint(original_model, checkpoint_path, threshold=0.375)
    reloaded_model, checkpoint = load_checkpoint(checkpoint_path)

    with torch.no_grad():
        reloaded_output = reloaded_model(inputs)
    torch.testing.assert_close(reloaded_output, original_output, rtol=0, atol=0)
    assert set(checkpoint) == {
        "model_state_dict",
        "architecture",
        "input_shape",
        "output_shape",
        "threshold",
        "num_classes",
    }
    assert checkpoint["input_shape"] == (3, 64, 64)
    assert checkpoint["output_shape"] == (1,)
