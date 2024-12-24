import atexit
from datetime import date
from functools import reduce
from io import BytesIO
from math import exp
from math import isnan as isnan_
from math import log
from threading import Lock, Thread
from typing import Any, Final, Iterable, NamedTuple, cast

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer
from matplotlib import use
from matplotlib.cm import plasma  # type: ignore
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import colorbar  # type: ignore
from matplotlib.pyplot import figure  # type: ignore
from matplotlib.pyplot import grid  # type: ignore
from matplotlib.pyplot import imshow  # type: ignore
from matplotlib.pyplot import legend  # type: ignore
from matplotlib.pyplot import plot  # type: ignore
from matplotlib.pyplot import savefig  # type: ignore
from matplotlib.pyplot import subplot  # type: ignore
from matplotlib.pyplot import title  # type: ignore
from matplotlib.pyplot import xlabel  # type: ignore
from matplotlib.pyplot import xticks  # type: ignore
from matplotlib.pyplot import ylabel  # type: ignore
from matplotlib.pyplot import yticks  # type: ignore
from matplotlib.pyplot import close, tight_layout
from networkx import draw  # type: ignore
from networkx import draw_networkx_labels  # type: ignore
from networkx import get_edge_attributes  # type: ignore
from networkx import spring_layout  # type: ignore
from networkx import Graph
from numpy import abs, array, concatenate, dtype
from numpy import float64 as float64_
from numpy import full, indices, nan, ndarray, newaxis, vstack, where
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, isnan
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from torch import from_numpy  # type: ignore
from torch import save  # type: ignore
from torch import Tensor, float32, float64, no_grad, zeros
from torch.nn import Dropout, Linear, Module, ModuleList, MSELoss
from torch.nn.functional import relu
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.nn import GATv2Conv  # type: ignore
from tqdm import tqdm
from typing_extensions import override

use("svg")


def xlogx(x: float) -> float:
    assert x >= 0
    return x if x <= 1 else 1 + log(x)


def map_(x: float) -> float:
    x = xlogx(x) if x >= 0 else -xlogx(-x)
    assert not isnan_(x)
    return x


def xexpx(x: float) -> float:
    assert x >= 0
    return x if x <= 1 else exp(x - 1)


def imap_(x: float):
    x = xexpx(x) if x >= 0 else -xexpx(-x)
    assert not isnan_(x)
    return x


class DataStore:

    def join_df(self, df: DataFrame):
        class_ids = sorted(
            df.select("CLASS_ID")
            .distinct()
            .rdd.flatMap(lambda x: cast(Iterable[str], x))
            .collect()
        )

        base_df = df.select("YEAR", "MONTH").distinct()
        for i in tqdm(range(0, len(class_ids), 10), desc="join table"):
            for class_id in class_ids[i : i + 10]:
                filtered_df = df.filter(col("CLASS_ID") == class_id).select(
                    "YEAR", "MONTH", col("UNITS").alias(f"UNITS_{class_id}")
                )
                base_df = base_df.join(filtered_df, on=["YEAR", "MONTH"], how="left")
            assert (
                base_df.filter(
                    reduce(
                        lambda a, b: a | b,
                        (
                            col(f"`UNITS_{c}`").isNull() | isnan(f"`UNITS_{c}`")
                            for c in class_ids[i : i + 10]
                        ),
                    )
                ).count()
                == 0
            )
        return class_ids, base_df

    def __init__(self) -> None:
        super().__init__()
        schema = StructType(
            [
                StructField("CLASS_ID", StringType(), False),  # not allowed NULL
                StructField("CHANNEL_ID", StringType(), False),
                StructField("YEAR", IntegerType(), False),
                StructField("MONTH", IntegerType(), False),
                StructField("UNITS", DoubleType(), False),
            ]
        )
        df = spark.read.csv(
            "file:///home/andy/EECSE6893/yf.csv", header=True, schema=schema
        )
        df.show()
        assert df.filter(df["CHANNEL_ID"] != "90.0").count() == 0
        class_ids, df = self.join_df(df)
        df = df.orderBy("YEAR", "MONTH")
        df.select(
            "YEAR",
            "MONTH",
            f"`UNITS_{class_ids[0]}`",
            f"`UNITS_{class_ids[1]}`",
            f"`UNITS_{class_ids[2]}`",
            f"`UNITS_{class_ids[3]}`",
            f"`UNITS_{class_ids[4]}`",
            f"`UNITS_{class_ids[5]}`",
        ).show()
        self.df = df
        self.class_ids = class_ids

    def months(self) -> tuple[date, ...]:
        return tuple(
            date(a, b, 1)
            for a, b in cast(
                Iterable[tuple[int, int]], self.df.select("YEAR", "MONTH").collect()
            )
        )


class DataSet:
    def __init__(
        self,
        data_store: DataStore,
        columns: tuple[str, ...],
        train_ratio: float,
        window_size: int,
        log_scale: bool = True,
    ) -> None:
        super().__init__()
        columns = tuple(c for c in columns if c not in ("YEAR", "MONTH"))
        df = data_store.df[[f"`{c}`" for c in columns]]
        count = df.count()
        if log_scale:
            df = df.rdd.map(
                lambda x: tuple(map_(y) for y in cast(Iterable[float], x))
            ).toDF(columns)
        train_count = round(count * train_ratio)
        train_df = df.limit(train_count)
        samples = train_df.rdd.flatMap(lambda x: cast(Iterable[float], x))
        mean = samples.mean()
        stdev = float(samples.stdev())

        df = df.rdd.map(
            lambda x: tuple((y - mean) / stdev for y in cast(Iterable[float], x))
        ).toDF(columns)

        self.data_store = data_store
        self.df = (
            df.rdd.zipWithIndex().map(lambda x: x[1:] + x[0]).toDF(("index",) + columns)
        )
        self.columns = columns
        self.log_scale = log_scale
        self.mean = mean
        self.stdev = stdev
        self.window_size = window_size
        self.train_count = train_count

    """
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor | None, bool]:
        assert i + self.window_size <= self.df.count()
        slice = self.df.filter(
            (i <= self.df["index"]) & (self.df["index"] < i + self.window_size)
        ).drop("index")
        (mean,) = slice.agg({f"`{c}`": "avg" for c in slice.columns}).collect()
        (stdev,) = slice.agg({f"`{c}`": "stddev" for c in slice.columns}).collect()
        normalized = as_tensor(
            slice.withColumns(
                {
                    f"`{c}`": (col(f"`{c}`") - mean[f"avg({c})"])
                    / stdev[f"stddev({c})"]
                    for c in slice.columns
                }
            )
            .fillna(0)
            .collect(),
            dtype=float64,
        )
        if i + self.window_size < self.df.count():
            y = as_tensor(
                self.df.filter(self.df["index"] == i + self.window_size)
                .drop("index")
                .collect(),
                dtype=float32,
            )
        else:
            y = None
        return (
            as_tensor(slice.collect(), dtype=float32),
            (normalized.T @ normalized).type(float32).abs(),
            y,
            i + self.window_size < self.train_count,
        )
    """

    def train(self) -> "DataLoaderAdapter":
        return DataLoaderAdapter(self, 0, self.train_count - self.window_size)

    def test(self) -> "DataLoaderAdapter":
        return DataLoaderAdapter(
            self,
            self.train_count - self.window_size,
            self.df.count() - self.window_size + 1,
        )


class Data(NamedTuple):
    x: Tensor
    y: Tensor
    e: Tensor
    w: Tensor


"""
class GAT(Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = Linear(4, 16, True)
        self.lin3 = Linear(16, 1, True)
        self.bn1_1 = BatchNorm1d(4)
        # self.bn1_2 = BatchNorm1d(16)
        self.bn2_1 = BatchNorm1d(16)
        # self.bn2_2 = BatchNorm1d(16)
        # self.bn3_1 = BatchNorm1d(16)
        self.bn3_2 = BatchNorm1d(16)
        self.gal1_1 = GATv2Conv(4, 4, heads=4, concat=True, edge_dim=1)
        # self.gal1_2 = GATv2Conv(16, 4, heads=4, concat=True, edge_dim=1)
        self.gal2_1 = GATv2Conv(16, 4, heads=4, concat=True, edge_dim=1)
        # self.gal2_2 = GATv2Conv(16, 4, heads=4, concat=True, edge_dim=1)
        # self.gal3_1 = GATv2Conv(16, 4, heads=4, concat=True, edge_dim=1)
        self.gal3_2 = GATv2Conv(16, 1, heads=1, concat=False, edge_dim=1)

    @override
    def forward(self, x: Tensor, e: Tensor, w: Tensor) -> Tensor:
        conv = self.gal1_1(relu(self.bn1_1(x)), e, w)
        # conv = self.gal1_2(relu(self.bn1_2(conv)), e, w)
        x = self.lin1(x) + conv
        conv = self.gal2_1(relu(self.bn2_1(x)), e, w)
        # conv = self.gal2_2(relu(self.bn2_2(conv)), e, w)
        x = x + conv
        # conv = self.gal3_1(relu(self.bn3_1(x)), e, w)
        conv = self.gal3_2(relu(self.bn3_2(conv)), e, w)
        return self.lin3(x) + conv
"""


class GAT(Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = Linear(4, 16, True)
        self.lin2 = Linear(16, 1, True)
        self.drop1 = Dropout()
        self.gal1 = GATv2Conv(4, 4, heads=4, concat=True, edge_dim=1)
        self.gal2 = GATv2Conv(16, 1, heads=1, concat=False, edge_dim=1)
        self.gals: Final[ModuleList] = ModuleList(
            (
                # GATv2Conv(16, 16, heads=4, concat=False, edge_dim=1),
                # GATv2Conv(16, 16, heads=4, concat=False, edge_dim=1),
                GATv2Conv(16, 4, heads=4, concat=True, edge_dim=1),
            )
        )
        self.drops: Final[ModuleList] = ModuleList(
            (
                Dropout(),
                # Dropout(),
                # Dropout(),
            )
        )
        # self.mask = ((torch.arange(0, 128)[newaxis, :] < 64) * 2 - 1).double().cuda()

    @override
    def forward(self, x: Tensor, e: Tensor, w: Tensor) -> Tensor:
        x = self.lin1(x) + self.drop1(relu(self.gal1(x, e, w)))  # * self.mask
        for gal, drop in zip(self.gals, self.drops):
            x = x + drop(relu(gal(x, e, w)))  # * self.mask
        x = self.lin2(x) + self.gal2(x, e, w)
        return x


class DataLoaderAdapter(list[Data], _Dataset[Data]):
    def make_A(
        self, data: ndarray[tuple[int, int], dtype[float64_]]
    ) -> ndarray[tuple[int, int], dtype[float64_]]:
        data = data - data.mean(axis=0, keepdims=True)
        std_ = data.std(axis=0, keepdims=True)
        std_[std_ == 0] = 1
        data = data / std_
        A = data.T @ data
        return abs(A / data.shape[0])

    def make_data(
        self,
        data: ndarray[tuple[int, int], dtype[float64_]],
        y: ndarray[tuple[int, int], dtype[float64_]],
    ) -> Data:
        A = self.make_A(data)
        x = from_numpy(data.T).type(float32).cuda()
        edge_weight = from_numpy(A[self.V][:, newaxis]).type(float32).cuda()
        return Data(x, from_numpy(y).type(float32).cuda(), self.edge_index, edge_weight)

    def __init__(self, data_set: DataSet, a: int, b: int) -> None:
        super().__init__()
        n = len(data_set.df.columns) - 1
        I, J = indices((n, n))
        self.V = where(I != J)
        self.edge_index = from_numpy(vstack((I[self.V], J[self.V]))).cuda()
        data = array(data_set.df.drop("index").collect(), dtype=float64_)
        for i in range(a, b):
            subs = data[i : i + data_set.window_size, :]
            if i + data_set.window_size < data_set.df.count():
                y = data[i + data_set.window_size, :, newaxis]
            else:
                y = full((data.shape[1], 1), nan)
            self.append(self.make_data(subs, y))


class Result(NamedTuple):
    train: tuple[tuple[float, ...], ...]
    test: tuple[tuple[float, ...], ...]
    pred: tuple[tuple[float, ...], ...]


class State:
    result: Result | None = None
    train_loss: list[float] = []
    test_loss: list[float] = []
    truth: tuple[tuple[float, ...], ...] = ()
    columns: tuple[str, ...] = ()
    lock = Lock()
    times: tuple[date, ...] = ()


def train(epochs: int = 4000):
    assert isinstance(train_loader.batch_size, int)
    model.train()
    loss = zeros((1,), dtype=float64, device="cuda")
    with SummaryWriter() as writer:
        for epoch in range(1, epochs + 1):
            if epoch == 1 or epoch % 100 == 0:
                save(model, f"model-{epoch}.pth")
            for data in train_loader:
                optimizer.zero_grad()
                loss = zeros((1,), dtype=float64, device="cuda")
                for i in range(data.x.shape[0]):
                    out = model(data.x[i], data.e[i], data.w[i])
                    loss = loss + criterion(out, data.y[i]) / train_loader.batch_size
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss = 0.0
            train_res: list[list[float]] = []
            with no_grad():
                for data in train_set:
                    out: Tensor = model(data.x, data.e, data.w)
                    train_res.append(
                        cast(list[float], out[:, 0].detach().cpu().tolist())
                    )
                    loss: Tensor = criterion(out, data.y)
                    train_loss += loss.item()
            train_loss /= len(train_set) - 1
            writer.add_scalar("Loss/train", train_loss, epoch)

            test_loss = 0.0
            test_res: list[list[float]] = []
            with no_grad():
                for data in test_set[:-1]:
                    out = model(data.x, data.e, data.w)
                    test_res.append(
                        cast(list[float], out[:, 0].detach().cpu().tolist())
                    )
                    loss: Tensor = criterion(out, data.y)
                    test_loss += loss.item()
            test_loss /= len(test_set) - 1

            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.flush()
            print(f"Epoch {epoch}, train_loss = {train_loss}, test_loss = {test_loss}")

            subs = truth[-data_set.window_size :, :]
            nans = full((subs.shape[1], 1), nan)
            with no_grad():
                for _ in range(data_set.window_size):
                    data = train_set.make_data(subs, nans)
                    out = model(data.x, data.e, data.w)[:, 0].cpu().numpy()
                    subs = concatenate((subs[1:, :], out[newaxis, :]))
            with state.lock:
                state.result = Result(
                    train=tuple(
                        tuple(imap_(data_set.mean + x * data_set.stdev) for x in y)
                        for y in train_res
                    ),
                    test=tuple(
                        tuple(imap_(data_set.mean + x * data_set.stdev) for x in y)
                        for y in test_res
                    ),
                    pred=tuple(
                        tuple(imap_(data_set.mean + x * data_set.stdev) for x in y)
                        for y in subs.tolist()
                    ),
                )
                state.train_loss.append(train_loss)
                state.test_loss.append(test_loss)


app = Flask(__name__, static_folder=".")


@app.route("/")
def web_index():
    return app.send_static_file("index.html")


@app.route("/draw_plot", methods=["POST"])
def draw_plot():
    with state.lock:
        if state.result is None:
            return jsonify({"img": ""})
        train_res = list(map(list, state.result.train))
        test_res = list(map(list, state.result.test))
        pred_res = list(map(list, state.result.pred))
    columns = set(request.get_json()["columns"])
    figure(figsize=(8, 4))
    n = len(train_res)
    m = n + len(test_res)
    lines: list[Any] = []
    for i, column in enumerate(state.columns):
        if f" {column[6:]} " not in columns:
            continue
        (p,) = plot(state.times[:n], [x[i] for x in state.truth[:n]], linewidth=1, label=column[6:])  # type: ignore
        lines.append(p.get_color())
    iter_ = iter(lines)
    for i, column in enumerate(state.columns):
        if f" {column[6:]} " not in columns:
            continue
        plot(state.times[n - 1 : m], [state.truth[n - 1][i]] + [x[i] for x in test_res], linewidth=1, linestyle="--", color=next(iter_))  # type: ignore
    iter_ = iter(lines)
    for i, column in enumerate(state.columns):
        if f" {column[6:]} " not in columns:
            continue
        plot(state.times[m - 1 :], [test_res[-1][i]] + [x[i] for x in pred_res], linewidth=1, linestyle=":", color=next(iter_))  # type: ignore
    legend(loc="lower center", prop={"family": "serif"}, ncols=5)
    grid(True)
    title("Truth vs Prediction", fontdict={"family": "serif"})
    xlabel("Month", fontdict={"family": "serif"})
    ylabel("Unit", fontdict={"family": "serif"})
    xticks(fontfamily="serif")
    yticks(fontfamily="serif")
    tight_layout()
    bio = BytesIO()
    savefig(bio, format="svg")
    close()
    bio.seek(0)
    content = bio.read().decode()
    content = content[content.find("<svg") :]
    return jsonify({"img": content})


@app.route("/status")
def get_state():
    with state.lock:
        return jsonify(
            {
                "result": (
                    None
                    if state.result is None
                    else {
                        "train": state.result.train,
                        "test": state.result.test,
                        "pred": state.result.pred,
                    }
                ),
                "train_loss": state.train_loss,
                "test_loss": state.test_loss,
                "truth": state.truth,
                "columns": [f" {x[6:]} " for x in state.columns],
                "times": [x.strftime("%Y-%m") for x in state.times],
            }
        )


@app.route("/draw_loss")
def draw_loss():
    with state.lock:
        train_loss = list(state.train_loss)
        test_loss = list(state.test_loss)
    figure(figsize=(8, 4))
    plot(train_loss, linewidth=2, label="Train Loss")
    plot(test_loss, linewidth=2, label="Test Loss")
    legend(prop={"family": "serif"})
    grid(True)
    title("Loss vs Step", fontdict={"family": "serif"})
    xlabel("Step", fontdict={"family": "serif"})
    ylabel("Loss", fontdict={"family": "serif"})
    xticks(fontfamily="serif")
    yticks(fontfamily="serif")
    tight_layout()
    bio = BytesIO()
    savefig(bio, format="svg")
    close()
    bio.seek(0)
    content = bio.read().decode()
    content = content[content.find("<svg") :]
    return jsonify({"img": content})


@app.route("/draw_heat", methods=["POST"])
def draw_heat():
    with state.lock:
        if state.result is None:
            return jsonify({"img": ""})
        pred_res = list(map(list, state.result.pred))
    A = train_set.make_A(
        array(
            [[(map_(x) - data_set.mean) / data_set.stdev for x in y] for y in pred_res]
        )
    )
    print((float(A.min()), float(A.max())))
    figure(figsize=(8, 4))
    subplot(1, 2, 1)
    im = imshow(A, cmap="plasma", vmin=0, vmax=1)
    cbar = colorbar(im, shrink=0.7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(FontProperties(family="serif"))
    xticks(fontfamily="serif")
    yticks(fontfamily="serif")
    xlabel("Class #", fontdict={"family": "serif"})
    ylabel("Class #", fontdict={"family": "serif"})
    title("Correlations", fontdict={"family": "serif"})
    columns: set[str] = set(request.get_json()["columns"])
    G = Graph()
    for column in state.columns:
        if f" {column[6:]} " in columns:
            G.add_node(column[6:])
    for i, c1 in enumerate(state.columns):
        if f" {c1[6:]} " in columns:
            for j, c2 in enumerate(state.columns[:i]):
                if f" {c2[6:]} " in columns:
                    G.add_edge(c1[6:], c2[6:], weight=A[i, j])
    pos: Any = spring_layout(G, weight="weight")
    weights: Any = get_edge_attributes(G, "weight").values()
    norm: Any = Normalize(vmin=0, vmax=1)
    edge_colors: Any = [plasma(norm(weight)) for weight in weights]
    subplot(1, 2, 2)
    title("Class Graph", fontdict={"family": "serif"})
    draw(
        G,
        pos,
        with_labels=False,
        node_color="lightblue",
        node_size=500,
        edge_color=edge_colors,
    )
    draw_networkx_labels(
        G,
        pos,
        {x: x for x in G.nodes()},  # type: ignore
        font_family="serif",
        font_size=10,
        font_color="k",
    )
    bio = BytesIO()
    savefig(bio, format="svg")
    close()
    bio.seek(0)
    content = bio.read().decode()
    content = content[content.find("<svg") :]
    return jsonify({"img": content})


if __name__ == "__main__":
    spark = cast(
        SparkSession,
        SparkSession.builder.appName("Connect to Standalone Spark Cluster")  # type: ignore
        .master("spark://localhost:7077")
        .getOrCreate(),
    )
    spark_ctx = spark.sparkContext
    atexit.register(lambda: spark.stop())
    data_store = DataStore()
    months = data_store.months()
    data_set = DataSet(data_store, tuple(data_store.df.columns), 0.7, 4, True)
    train_set = data_set.train()
    train_loader = DataLoader(train_set)
    test_set = data_set.test()
    model = GAT().type(float32).cuda()
    data = next(iter(train_loader))
    out = model(data.x[0].cuda(), data.e[0].cuda(), data.w[0].cuda())
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    criterion = MSELoss()
    state = State()
    truth = array(data_set.df.drop("index").collect(), dtype=float64_)
    state.truth = tuple(
        tuple(
            imap_(data_set.mean + x * data_set.stdev) for x in cast(Iterable[float], y)
        )
        for y in truth[data_set.window_size :]
    )
    state.times = data_store.months()[data_set.window_size :]
    for _ in range(data_set.window_size):
        state.times += (
            date(
                year=state.times[-1].year + state.times[-1].month // 12,
                month=state.times[-1].month % 12 + 1,
                day=1,
            ),
        )
    state.columns = data_set.columns
    thread = Thread(target=train)
    thread.start()
    http_server = WSGIServer(("0.0.0.0", 80), app)
    http_server.serve_forever()
