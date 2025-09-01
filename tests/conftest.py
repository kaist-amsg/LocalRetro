import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add the project root to sys.path so imports work during testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file within the temp directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "model_config": {
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.1
        },
        "training_config": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        },
        "data_config": {
            "dataset": "USPTO_50K",
            "data_dir": "./data"
        }
    }


@pytest.fixture
def mock_config_file(temp_dir, mock_config):
    """Create a temporary config JSON file."""
    config_file = temp_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(mock_config, f, indent=2)
    return config_file


@pytest.fixture
def sample_molecule_smiles():
    """Provide sample SMILES strings for testing."""
    return [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
        "CCN(CC)CC",  # triethylamine
        "C1CCOC1"  # tetrahydrofuran
    ]


@pytest.fixture
def sample_reaction_smiles():
    """Provide sample reaction SMILES for testing."""
    return [
        "CCO>>CC=O",  # oxidation
        "c1ccccc1Br.CCO>>c1ccccc1OCC",  # etherification
        "CC(=O)O.CCO>>CC(=O)OCC",  # esterification
    ]


@pytest.fixture
def mock_rdkit_mol():
    """Provide a mock RDKit molecule object."""
    mock_mol = Mock()
    mock_mol.GetNumAtoms.return_value = 5
    mock_mol.GetNumBonds.return_value = 4
    mock_mol.GetAtoms.return_value = [Mock() for _ in range(5)]
    mock_mol.GetBonds.return_value = [Mock() for _ in range(4)]
    return mock_mol


@pytest.fixture
def mock_torch_tensor():
    """Provide a mock PyTorch tensor."""
    mock_tensor = Mock()
    mock_tensor.shape = (32, 128)
    mock_tensor.size.return_value = (32, 128)
    mock_tensor.dtype = "torch.float32"
    return mock_tensor


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    mock_model = Mock()
    mock_model.train.return_value = None
    mock_model.eval.return_value = None
    mock_model.forward.return_value = Mock()
    mock_model.parameters.return_value = [Mock() for _ in range(10)]
    return mock_model


@pytest.fixture
def mock_dataloader():
    """Provide a mock dataloader for testing."""
    mock_batch = {
        'molecules': Mock(),
        'templates': Mock(),
        'labels': Mock()
    }
    mock_dataloader = Mock()
    mock_dataloader.__iter__.return_value = [mock_batch] * 5
    mock_dataloader.__len__.return_value = 5
    return mock_dataloader


@pytest.fixture
def sample_template_data():
    """Provide sample template data for testing."""
    return {
        "atom_templates": [
            {"template_id": 1, "template": "[C:1]", "count": 100},
            {"template_id": 2, "template": "[O:1]", "count": 80},
            {"template_id": 3, "template": "[N:1]", "count": 60}
        ],
        "bond_templates": [
            {"template_id": 1, "template": "[C:1]-[C:2]", "count": 150},
            {"template_id": 2, "template": "[C:1]-[O:2]", "count": 120},
            {"template_id": 3, "template": "[C:1]=[O:2]", "count": 90}
        ]
    }


@pytest.fixture
def mock_dgl_graph():
    """Provide a mock DGL graph object."""
    mock_graph = Mock()
    mock_graph.number_of_nodes.return_value = 10
    mock_graph.number_of_edges.return_value = 12
    mock_graph.ndata = {'feat': Mock()}
    mock_graph.edata = {'feat': Mock()}
    return mock_graph


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide the path to test data directory."""
    test_data_path = Path(__file__).parent / "data"
    test_data_path.mkdir(exist_ok=True)
    return test_data_path


@pytest.fixture
def caplog_debug(caplog):
    """Set up logging capture at DEBUG level."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def mock_environment_variables():
    """Provide mock environment variables."""
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/workspace",
        "MODEL_DIR": "/workspace/models"
    }
    
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up imports after each test to avoid module pollution."""
    # Store original modules
    original_modules = set(sys.modules.keys())
    
    yield
    
    # Remove any modules that were imported during the test
    for module_name in list(sys.modules.keys()):
        if module_name not in original_modules:
            if (module_name.startswith('LocalTemplate') or 
                module_name.startswith('scripts')):
                del sys.modules[module_name]