import os
import pytest
import numpy as np

from schnetpack.datasets import QM9, MD17, rMD17
import schnetpack.transform as trn

@pytest.fixture
def test_qm9_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")
    return path


@pytest.mark.skip(
    "Run only local, not in CI. Otherwise takes too long and requires downloading "
    + "the data"
)
def test_qm9(test_qm9_path):
    qm9 = QM9(
        test_qm9_path,
        num_train=10,
        num_val=5,
        batch_size=5,
        remove_uncharacterized=True,
    )
    assert len(qm9.train_dataset) == 10
    assert len(qm9.val_dataset) == 5
    assert len(qm9.test_dataset) == 5

    ds = [b for b in qm9.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in qm9.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in qm9.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_md17_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_md17.db")
    return path


@pytest.mark.skip(
    "Run only local, not in CI. Otherwise takes too long and requires downloading "
    + "the data"
)
def test_md17(test_md17_path):
    md17 = MD17(
        test_md17_path,
        num_train=10,
        num_val=5,
        num_test=5,
        batch_size=5,
        molecule="uracil",
    )
    md17.prepare_data()
    md17.setup()
    assert len(md17.train_dataset) == 10
    assert len(md17.val_dataset) == 5
    assert len(md17.test_dataset) == 5

    ds = [b for b in md17.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in md17.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in md17.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_rmd17_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_rmd17.db")
    return path


@pytest.mark.skip(
    "Run only local, not in CI. Otherwise takes too long and requires downloading "
    + "the data"
)
def test_rmd17(test_rmd17_path):
    md17 = rMD17(
        test_rmd17_path,
        num_train=950,
        num_val=50,
        num_test=1000,
        batch_size=5,
        molecule="uracil",
    )
    md17.prepare_data()
    md17.setup()
    assert len(md17.train_dataset) == 950
    assert len(md17.val_dataset) == 50
    assert len(md17.test_dataset) == 1000

    train_idx = md17.train_dataset.subset_idx
    val_idx = md17.val_dataset.subset_idx
    test_idx = md17.test_dataset.subset_idx
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0



@pytest.fixture
def test_hopv15_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_hopv15.db")
    return path


# @pytest.mark.skip(
#     "Run only local, not in CI. Otherwise takes too long and requires downloading "
#     + "the data"
# )
def test_hopv15(test_hopv15_path):
    from schnetpack.datasets import HOPV15

    hopv15 = HOPV15(
        test_hopv15_path,
        num_train=10,
        num_val=5,
        num_test=5,
        batch_size=5,
    )
    hopv15.prepare_data()
    hopv15.setup()
    assert len(hopv15.train_dataset) == 10
    assert len(hopv15.val_dataset) == 5
    assert len(hopv15.test_dataset) == 5

    ds = [b for b in hopv15.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in hopv15.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in hopv15.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_naphtha_path():
    path = os.path.join(os.path.dirname(__file__), "../testdata/tmp/test_naphtha.db")
    return path


# @pytest.mark.skip(
#     "Run only local, not in CI. Otherwise takes too long and requires downloading "
#     + "the data"
# )
def test_naphtha(test_naphtha_path):
    from schnetpack.datasets import Naphtha

    naphtha = Naphtha(
        test_naphtha_path,
        xyz_path="/Users/masoud/Projects/schnetpack/examples/naphtha/split_xyz_files",
        num_train=10,
        num_val=5,
        num_test=5,
        batch_size=5,
        train_transforms=[trn.KNNRepresentation(k=3, threshold=2.0)],
        val_transforms=[trn.KNNRepresentation(k=3, threshold=2.0)],
        test_transforms=[trn.KNNRepresentation(k=3, threshold=2.0)],
    )
    naphtha.prepare_data()
    naphtha.setup()
    assert len(naphtha.train_dataset) == 10
    assert len(naphtha.val_dataset) == 5
    assert len(naphtha.test_dataset) == 5

    ds = [b for b in naphtha.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in naphtha.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in naphtha.test_dataloader()]
    assert len(ds) == 1