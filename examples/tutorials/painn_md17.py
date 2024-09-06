# Block 1

import os
import schnetpack as spk
from schnetpack.datasets import MD17
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
import wandb

data_path = './md17'
if not os.path.exists('md17'):
    os.makedirs(data_path)

# Block 1
def calculate_mean_and_variance_labels(dataloader):
    all_labels = []
    for batch in dataloader:
        all_labels.append(batch[MD17.energy])  # Assuming the labels are stored in MD17.energy
    all_labels = torch.cat(all_labels, dim=0)

    mean = torch.mean(all_labels, dim=0)
    variance = torch.var(all_labels, dim=0)

    return mean, variance


# Block 2

md17data = MD17(
    os.path.join(data_path, 'ethanol.db'),
    molecule='ethanol',
    batch_size=10,
    num_train=1000,
    num_val=1000,
    num_test=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=2,
    pin_memory=True,
)
md17data.prepare_data()
md17data.setup()

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
pred_G = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)

nnpot = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[pairwise_distance],
    output_modules=[
        pred_G
    ],
    postprocessors=[trn.CastTo64(),
                    trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=True),
                    ]
)

output_G = spk.task.ModelOutput(
    name=MD17.energy,
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
    save_dir=data_path,
    name=run_name,
    version=0)
wandb.init(project="PaiNN", sync_tensorboard=True)

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(data_path, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=data_path,
    max_epochs=100,
)
trainer.fit(task, datamodule=md17data)
# model = spk.representation.PaiNN(
#     n_atom_basis=n_atom_basis, n_interactions=msg_passing,
#     radial_basis=radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(cutoff)
# )
# ckpt = torch.load("qm9tut/G-MSGPass3-atomBasisNumber30/version_0/checkpoints/epoch=96-step=9700.ckpt")
# model.load_state_dict(ckpt['optimizer_state_dict'])
# model = spk.representation.PaiNN.load_from_checkpoint("checkpoints/epoch=96-step=9700.ckpt")
trainer.test(task, md17data.test_dataloader())
