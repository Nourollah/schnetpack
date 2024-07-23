# Block 1

import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
import wandb

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

# Block 2

qm9data = QM9(
    './qm9.db',
    batch_size=100,
    num_train=20000,
    num_val=1001,
    transforms=[
        trn.KNNRepresentation(k=3, threshold=5.),
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(QM9.G, remove_mean=True, remove_atomrefs=True),
        trn.CastTo32()
    ],
    property_units={
        QM9.G: 'Ha',
    },
    num_workers=2,

    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True,  # set to false, when not using a GPU
    remove_uncharacterized=True,
    load_properties=[
        QM9.G,
    ],  # only load U0 property
)
qm9data.prepare_data()
qm9data.setup()

# Block 3

msg_passing = 3
cutoff = 5.
n_atom_basis = 30

pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
painn = spk.representation.PaiNN(
    n_atom_basis=n_atom_basis, n_interactions=msg_passing,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_G = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.G)

nnpot = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[
        pred_G
    ],
    postprocessors=[trn.CastTo64(),
                    trn.AddOffsets(QM9.G, add_mean=True, add_atomrefs=True),
                    ]
)

output_G = spk.task.ModelOutput(
    name=QM9.G,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

# Block 4
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[
        output_G
    ],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# Block 5
run_name = f"G-MSGPass{msg_passing}-atomBasisNumber{n_atom_basis}"
logger = pl.loggers.TensorBoardLogger(
    save_dir=qm9tut,
    name=run_name,
    version=0)
wandb.init(project="PaiNN", sync_tensorboard=True)

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(qm9tut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=1000,  # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=qm9data)
