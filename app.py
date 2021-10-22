import streamlit as st

from gen import utils, data, ae, vae
from fastai.tabular.all import *

"""
# Playing with generative models
"""

model_type = st.sidebar.radio("Pick the model", ["AE", "VAE"], index=1)
data_set = st.sidebar.radio(
    "Pick the dataset",
    [
        "twospirals",
        "twomoons",
        "sign",
        "abs",
        "sinewave",
        "crescentcube",
        "crescent",
        "gaussian",
        "checkerboard",
    ],
)
n_data = st.sidebar.number_input("Number of samples", value=4_200, min_value=100)
n_epochs = st.sidebar.number_input("Number of epochs", value=10, min_value=1)
kld_weight = st.sidebar.number_input("KLD weight", value=0.001, min_value=0.0)
p_drop = st.sidebar.number_input(
    "Dropout chance", value=0.01, min_value=0.0001, max_value=0.99
)
show_loss = st.sidebar.checkbox("Show loss")
show_report = st.sidebar.checkbox("Show report")

model_name_map = {
    "AE": "an auto encoder (AE)",
    "VAE": "a variational auto encoder (VAE)",
}


def run_stuff(model_type, data_set, n, epochs, kld_weight, p):
    assert model_type in ["VAE", "AE"]
    # generating data
    n = int(n)
    epochs = int(epochs)

    st.markdown(f"## Using {model_name_map[model_type]} model")

    y_col = "target"
    all_train_data = data.DataGenerator.generate(data_set, n).assign(
        **{y_col: np.random.choice([0, 1], size=n)}
    )

    # pre-processing data
    splits = RandomSplitter(valid_pct=0.2)(all_train_data)

    original_features = L(
        [c for c in all_train_data.columns if c != "id" and c != y_col]
    )

    to = TabularPandas(
        all_train_data,
        procs=[FillMissing, Normalize],
        cont_names=original_features,
        y_names=y_col,
        splits=splits,
    )

    bs = 256
    dls = to.dataloaders(bs=bs)

    # setting up the model
    if model_type == "VAE":
        model = vae.VAE(n_in=len(original_features), n_h=200, n_z=2, p=p)
        loss_func = vae.VAE_Loss(kld_weight=kld_weight)  # bs
    else:
        model = ae.AE(n_in=len(original_features), n_h=200, n_z=2, p=p)
        loss_func = ae.AE_Loss()

    # training
    learn = Learner(dls, model, loss_func=loss_func)
    lrs = learn.lr_find()
    learn.fit_one_cycle(epochs, lr_max=lrs.valley)

    # inspecting model generated data
    if model_type == "VAE":
        (ori, rec, mu, logvar), _ = learn.get_preds(ds_idx=1)
    else:
        (ori, rec), _ = learn.get_preds(ds_idx=1)

    fig, report = utils.check_identifiability_of_generated_data(
        ori, rec, original_features
    )
    st.write("Comparing original training data and model guesses")
    st.pyplot(fig)

    if show_loss:
        st.write("Model loss")
        if model_type == "VAE":
            loss = loss_func.loss(rec, ori, mu, logvar)
        else:
            loss = loss_func.loss(rec, ori)

        loss["loss"] = loss["loss"].detach().item()
        loss["Reconstruction_Loss"] = loss["Reconstruction_Loss"].detach().item()
        if "KLD" in loss:
            loss["KLD"] = loss["KLD"].detach().item()
        st.dataframe(data=pd.Series(loss).to_frame().T)

    if show_report:
        st.write(
            "Classification report for another model trying to distinguish between model generated and original data:"
        )
        st.dataframe(
            data=pd.DataFrame({k: v for k, v in report.items() if k != "accuracy"})
        )
        st.write(f'Accuracy = {report["accuracy"]}')


if __name__ == "__main__":
    run_stuff(model_type, data_set, n_data, n_epochs, kld_weight, p_drop)
