from loguru import logger
import pandas as pd
from torch import Tensor
import matplotlib.pyplot as plt
from fastai.tabular.all import RandomSplitter
import lightgbm as lgbm
from sklearn import metrics


class LogFilter:
    """Filter for loguru logger

    Example to show everything from debug on:
    log_filter = LogFilter("DEBUG") # "WARNING"
    logger.remove()
    logger.add(sys.stderr, filter=log_filter, level=0)
    """

    def __init__(self, level):
        self.level = level

    def __call__(self, record):
        levelno = logger.level(self.level).no
        return record["level"].no >= levelno


def check_identifiability_of_generated_data(
    ori: Tensor, rec: Tensor, original_features: list = ["x_0", "x_1"]
):
    logger.info(
        "Inspecting identifability of `ori` (original data) from `rec` (reconstructed data) using a classifier."
    )

    df_ori = pd.DataFrame(ori.detach().numpy(), columns=original_features)
    df_rec = pd.DataFrame(rec.detach().numpy(), columns=original_features)

    fig, ax = plt.subplots()
    ax.scatter(df_ori["x_0"], df_ori["x_1"], alpha=0.1, label="original")
    ax.scatter(df_rec["x_0"], df_rec["x_1"], alpha=0.1, label="model")
    ax.set(title="Original and reconstructed as returned by the model")
    ax.legend()
    # plt.show()

    df_vae = pd.concat(
        (
            pd.DataFrame(ori.detach(), columns=original_features).assign(set=0),
            pd.DataFrame(rec.detach(), columns=original_features).assign(set=1),
        ),
        ignore_index=True,
    )

    dis_splits = RandomSplitter(valid_pct=0.2)(df_vae)

    dis_model = lgbm.LGBMClassifier()

    dis_model.fit(
        df_vae.iloc[dis_splits[0]][original_features], df_vae.iloc[dis_splits[0]]["set"]
    )
    report = metrics.classification_report(
        df_vae.iloc[dis_splits[1]]["set"],
        dis_model.predict(df_vae.iloc[dis_splits[1]][original_features]),
        output_dict=True,
    )

    return fig, report
