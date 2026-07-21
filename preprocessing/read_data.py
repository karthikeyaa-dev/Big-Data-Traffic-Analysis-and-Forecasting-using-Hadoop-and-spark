from __future__ import annotations

import pickle
from io import StringIO
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
from pandas.io.pytables import TableIterator


class Data:
    """Utility class for reading METR-LA HDF5 and graph pickle datasets."""

    def __init__(
        self,
        h5_path: str | Path,
        pkl_path: str | Path,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.pkl_path = Path(pkl_path)

        self._h5_df: pd.DataFrame | None = None
        self._pkl_df: pd.DataFrame | None = None

    @property
    def h5_dataframe(self) -> pd.DataFrame:
        """Return loaded HDF5 dataframe."""
        if self._h5_df is None:
            raise ValueError("No HDF5 DataFrame loaded. Call read_h5() first.")

        return self._h5_df

    @property
    def pkl_dataframe(self) -> pd.DataFrame:
        """Return loaded pickle dataframe."""
        if self._pkl_df is None:
            raise ValueError("No pickle DataFrame loaded. Call read_pickle() first.")

        return self._pkl_df

    @overload
    def read_h5(
        self,
        *,
        key: str = "df",
        iterator: Literal[False] = False,
        chunksize: int | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def read_h5(
        self,
        *,
        key: str = "df",
        iterator: Literal[True],
        chunksize: int,
    ) -> TableIterator: ...

    def read_h5(
        self,
        *,
        key: str = "df",
        iterator: bool = False,
        chunksize: int | None = None,
    ) -> pd.DataFrame | TableIterator:
        """Read HDF5 traffic data."""

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        if iterator and chunksize is None:
            raise ValueError("chunksize must be specified when iterator=True.")

        if not iterator and chunksize is not None:
            raise ValueError("chunksize can only be specified when iterator=True.")

        try:
            result = pd.read_hdf(
                self.h5_path,
                key=key,
                iterator=iterator,
                chunksize=chunksize,
            )

        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to read HDF5 file '{self.h5_path}': {exc}"
            ) from exc

        if isinstance(result, TableIterator):
            return result

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Unexpected return type from pandas.read_hdf(): "
                f"{type(result).__name__}"
            )

        if isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.DatetimeIndex(result.index.values)

        self._h5_df = result

        return result

    def read_pickle(self) -> pd.DataFrame:
        """Read graph adjacency matrix from pickle file."""

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.pkl_path}")

        with open(self.pkl_path, "rb") as file:
            data = pickle.load(file, encoding="latin1")

        sensor_ids = data[0]
        adjacency_matrix = data[2]

        result = pd.DataFrame(
            adjacency_matrix,
            index=sensor_ids,
            columns=sensor_ids,
        )

        self._pkl_df = result

        return result

    def info(self) -> str:
        """Return information about loaded HDF5 dataframe."""

        if self._h5_df is None:
            raise ValueError("No HDF5 DataFrame loaded. Call read_h5() first.")

        buffer = StringIO()

        self._h5_df.info(buf=buffer)

        return buffer.getvalue()

    def save_parquet(
        self,
        df: pd.DataFrame,
        path: str | Path,
        *,
        engine: Literal[
            "auto",
            "pyarrow",
            "fastparquet",
        ] = "pyarrow",
        compression: Literal[
            "snappy",
            "gzip",
            "brotli",
            "lz4",
            "zstd",
        ]
        | None = "snappy",
        index: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Save dataframe as parquet file."""

        df.to_parquet(
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"h5_path={self.h5_path!r}, "
            f"pkl_path={self.pkl_path!r})"
        )


def main() -> None:
    data = Data(
        "../data/METR-LA.h5",
        "../data/adj_METR-LA.pkl",
    )

    # Load traffic speed data
    traffic_df = data.read_h5()

    print("Traffic Data:")
    print(traffic_df.head())

    print("\nTraffic Shape:")
    print(traffic_df.shape)

    graph_df = data.read_pickle()

    print("\nGraph Data:")
    print(graph_df.head())

    print("\nGraph Shape:")
    print(graph_df.shape)

    data.save_parquet(
        traffic_df,
        "../data/traffic.parquet",
    )

    data.save_parquet(
        graph_df,
        "../data/graph.parquet",
    )


if __name__ == "__main__":
    main()
