from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    MinMaxScaler,
)

from pyspark.ml import Pipeline

import logging

logger = logging.getLogger(__name__)


class TrafficFeatureBuilder:
    """
    Builds traffic forecasting features from raw sensor speed data.

    Expected input columns:
        sensor_id
        timestamp
        speed
    """

    def __init__(
        self,
        free_flow_speed: int = 60,
        moderate_speed: int = 40,
    ):
        self.free_flow_speed = free_flow_speed
        self.moderate_speed = moderate_speed

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_input(self, df: DataFrame) -> None:
        required = {
            "sensor_id",
            "timestamp",
            "speed",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------------------
    # Time Features
    # ---------------------------------------------------------

    def create_time_features(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return (
            df.withColumn("date", F.to_date("timestamp"))
            .withColumn("hour", F.hour("timestamp"))
            .withColumn("minute", F.minute("timestamp"))
            .withColumn("day_of_week", F.dayofweek("timestamp"))
            .withColumn("day_of_month", F.dayofmonth("timestamp"))
            .withColumn("week_of_year", F.weekofyear("timestamp"))
            .withColumn("month", F.month("timestamp"))
            .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]).cast("int"))
            .withColumn(
                "is_peak_hour",
                (
                    ((F.col("hour") >= 7) & (F.col("hour") <= 9))
                    | ((F.col("hour") >= 16) & (F.col("hour") <= 18))
                ).cast("int"),
            )
            .withColumn(
                "time_of_day",
                F.when(
                    F.col("hour").between(5, 11),
                    "Morning",
                )
                .when(
                    F.col("hour").between(12, 16),
                    "Afternoon",
                )
                .when(
                    F.col("hour").between(17, 20),
                    "Evening",
                )
                .otherwise("Night"),
            )
        )

    # ---------------------------------------------------------
    # Time Series Features
    # ---------------------------------------------------------

    def create_time_series_features(
        self,
        df: DataFrame,
    ) -> DataFrame:

        window = Window.partitionBy("sensor_id").orderBy("timestamp")

        windows = {
            3: window.rowsBetween(-2, 0),
            6: window.rowsBetween(-5, 0),
            12: window.rowsBetween(-11, 0),
        }

        result = (
            df.withColumn("lag_1", F.lag("speed", 1).over(window))
            .withColumn("lag_2", F.lag("speed", 2).over(window))
            .withColumn("lag_3", F.lag("speed", 3).over(window))
            .withColumn("lag_6", F.lag("speed", 6).over(window))
            .withColumn("lag_12", F.lag("speed", 12).over(window))
            .withColumn("speed_change", F.col("speed") - F.col("lag_1"))
        )

        for size, w in windows.items():
            result = (
                result.withColumn(f"rolling_mean_{size}", F.avg("speed").over(w))
                .withColumn(f"rolling_std_{size}", F.stddev("speed").over(w))
                .withColumn(f"rolling_max_{size}", F.max("speed").over(w))
                .withColumn(f"rolling_min_{size}", F.min("speed").over(w))
            )

        return result

    # ---------------------------------------------------------
    # Traffic Conditions
    # ---------------------------------------------------------

    def create_condition_features(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return (
            df.withColumn(
                "congestion_level",
                F.when(
                    F.col("speed") > self.free_flow_speed,
                    "Free Flow",
                )
                .when(
                    F.col("speed") >= self.moderate_speed,
                    "Moderate",
                )
                .otherwise("Congested"),
            )
            .withColumn(
                "speed_category",
                F.when(
                    F.col("speed") < self.moderate_speed,
                    "Low",
                )
                .when(
                    F.col("speed") <= self.free_flow_speed,
                    "Medium",
                )
                .otherwise("High"),
            )
            .withColumn(
                "traffic_trend",
                F.when(
                    F.col("speed_change") > 0,
                    "Increasing",
                )
                .when(
                    F.col("speed_change") < 0,
                    "Decreasing",
                )
                .otherwise("Stable"),
            )
            .withColumn(
                "percent_change",
                F.when(
                    F.col("lag_1") != 0,
                    ((F.col("speed") - F.col("lag_1")) / F.col("lag_1")) * 100,
                ),
            )
        )

    # ---------------------------------------------------------
    # Cyclic Encoding
    # ---------------------------------------------------------

    def create_cyclic_features(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return (
            df.withColumn("hour_sin", F.sin(2 * F.pi() * F.col("hour") / 24))
            .withColumn("hour_cos", F.cos(2 * F.pi() * F.col("hour") / 24))
            .withColumn("day_sin", F.sin(2 * F.pi() * (F.col("day_of_week") - 1) / 7))
            .withColumn("day_cos", F.cos(2 * F.pi() * (F.col("day_of_week") - 1) / 7))
            .withColumn("month_sin", F.sin(2 * F.pi() * (F.col("month") - 1) / 12))
            .withColumn("month_cos", F.cos(2 * F.pi() * (F.col("month") - 1) / 12))
        )

    # ---------------------------------------------------------
    # Sensor Statistics
    # ---------------------------------------------------------

    def create_sensor_statistics(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return df.groupBy("sensor_id").agg(
            F.avg("speed").alias("average_speed"),
            F.max("speed").alias("maximum_speed"),
            F.min("speed").alias("minimum_speed"),
            F.expr("percentile_approx(speed,0.5)").alias("median_speed"),
            F.stddev("speed").alias("standard_deviation"),
            F.variance("speed").alias("variance"),
        )

    # ---------------------------------------------------------
    # Create Prediction Target
    # ---------------------------------------------------------

    def create_target(
        self,
        df: DataFrame,
    ) -> DataFrame:

        window = Window.partitionBy("sensor_id").orderBy("timestamp")

        return df.withColumn("target_speed", F.lead("speed", 1).over(window)).dropna(
            subset=["target_speed"]
        )

    # ---------------------------------------------------------
    # Encode Categorical Features
    # ---------------------------------------------------------

    def encode_categorical(
        self,
        df: DataFrame,
    ):

        categorical_cols = [
            "time_of_day",
            "congestion_level",
            "speed_category",
            "traffic_trend",
        ]

        stages = []

        # String Indexing
        for c in categorical_cols:
            indexer = StringIndexer(
                inputCol=c, outputCol=f"{c}_index", handleInvalid="keep"
            )
            stages.append(indexer)

        # One Hot Encoding
        for c in categorical_cols:
            encoder = OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vector")
            stages.append(encoder)

        pipeline = Pipeline(stages=stages)

        model = pipeline.fit(df)

        return model.transform(df)

    # ---------------------------------------------------------
    # Normalize Features
    # ---------------------------------------------------------

    def normalize_features(
        self,
        df: DataFrame,
    ):

        exclude = [
            "sensor_id",
            "timestamp",
            "date",
            "speed",
            "target_speed",
            "time_of_day",
            "congestion_level",
            "speed_category",
            "traffic_trend",
            "time_of_day_index",
            "congestion_level_index",
            "speed_category_index",
            "traffic_trend_index",
            "time_of_day_vector",
            "congestion_level_vector",
            "speed_category_vector",
            "traffic_trend_vector",
        ]

        numeric_cols = [c for c in df.columns if c not in exclude]

        # Fill NULL values before assembling

        df = df.fillna(0, subset=numeric_cols)

        assembler = VectorAssembler(
            inputCols=numeric_cols, outputCol="numeric_features", handleInvalid="keep"
        )

        scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_features")

        pipeline = Pipeline(stages=[assembler, scaler])

        model = pipeline.fit(df)

        return model.transform(df)

    # ---------------------------------------------------------
    # Combine Features For GNN
    # ---------------------------------------------------------

    def create_gnn_features(
        self,
        df: DataFrame,
    ):

        assembler = VectorAssembler(
            inputCols=[
                "scaled_features",
                "time_of_day_vector",
                "congestion_level_vector",
                "speed_category_vector",
                "traffic_trend_vector",
            ],
            outputCol="features",
        )

        return assembler.transform(df)

    # ---------------------------------------------------------
    # Complete Pipeline
    # ---------------------------------------------------------

    def build_features(
        self,
        df: DataFrame,
    ) -> DataFrame:

        logger.info("Starting traffic feature generation")

        self.validate_input(df)

        # Time features
        df = self.create_time_features(df)

        # Lag + rolling features
        df = self.create_time_series_features(df)

        # Traffic labels
        df = self.create_condition_features(df)

        # Cyclic features
        df = self.create_cyclic_features(df)

        # Create future target
        df = self.create_target(df)

        # Encode categorical
        df = self.encode_categorical(df)

        # Normalize numerical
        df = self.normalize_features(df)

        # Final GNN vector
        df = self.create_gnn_features(df)

        logger.info("Traffic feature generation completed")

        return df
