import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark import StorageLevel


logger = logging.getLogger(__name__)


class TrafficGraphFeatureBuilder:
    """
    Generates graph-based traffic sensor features.

    Expected graph_df columns:
        source_sensor
        destination_sensor
        weight
    """

    def validate_input(
        self,
        graph_df: DataFrame,
    ) -> None:

        required = {
            "source_sensor",
            "destination_sensor",
            "weight",
        }

        missing = required - set(graph_df.columns)

        if missing:
            raise ValueError(f"Missing graph columns: {missing}")

    def create_degree_features(
        self,
        graph_df: DataFrame,
    ) -> DataFrame:

        outgoing = (
            graph_df.groupBy("source_sensor")
            .agg(F.count("*").alias("out_degree"))
            .withColumnRenamed(
                "source_sensor",
                "sensor_id",
            )
        )

        incoming = (
            graph_df.groupBy("destination_sensor")
            .agg(F.count("*").alias("in_degree"))
            .withColumnRenamed(
                "destination_sensor",
                "sensor_id",
            )
        )

        return (
            outgoing.join(
                incoming,
                "sensor_id",
                "outer",
            )
            .fillna(
                {
                    "out_degree": 0,
                    "in_degree": 0,
                }
            )
            .withColumn("degree", F.col("in_degree") + F.col("out_degree"))
        )

    def create_weight_features(
        self,
        graph_df: DataFrame,
    ) -> DataFrame:

        outgoing = (
            graph_df.groupBy("source_sensor")
            .agg(F.sum("weight").alias("out_weight_degree"))
            .withColumnRenamed(
                "source_sensor",
                "sensor_id",
            )
        )

        incoming = (
            graph_df.groupBy("destination_sensor")
            .agg(F.sum("weight").alias("in_weight_degree"))
            .withColumnRenamed(
                "destination_sensor",
                "sensor_id",
            )
        )

        return (
            outgoing.join(
                incoming,
                "sensor_id",
                "outer",
            )
            .fillna(
                {
                    "out_weight_degree": 0.0,
                    "in_weight_degree": 0.0,
                }
            )
            .withColumn(
                "weighted_degree",
                F.col("in_weight_degree") + F.col("out_weight_degree"),
            )
        )

    def create_neighbor_features(
        self,
        graph_df: DataFrame,
    ) -> DataFrame:

        return (
            graph_df.groupBy("source_sensor")
            .agg(
                F.count_distinct("destination_sensor").alias("neighbor_count"),
                F.avg("weight").alias("avg_neighbor_weight"),
            )
            .withColumnRenamed(
                "source_sensor",
                "sensor_id",
            )
        )

    def create_pagerank_features(
        self,
        graph_df: DataFrame,
        max_iter: int = 10,
        reset_probability: float = 0.15,
    ) -> DataFrame:

        logger.info("Calculating PageRank")

        vertices = (
            graph_df.select(F.col("source_sensor").alias("sensor_id"))
            .union(graph_df.select(F.col("destination_sensor").alias("sensor_id")))
            .distinct()
        )

        # Initial rank
        ranks = vertices.withColumn("pagerank", F.lit(1.0))

        outgoing_weight = graph_df.groupBy("source_sensor").agg(
            F.sum("weight").alias("total_weight")
        )

        edges = graph_df.join(outgoing_weight, "source_sensor", "left").select(
            F.col("source_sensor").alias("src"),
            F.col("destination_sensor").alias("dst"),
            (F.col("weight") / F.col("total_weight")).alias("transition"),
        )

        for _ in range(max_iter):
            contributions = (
                edges.join(
                    ranks,
                    edges.src == ranks.sensor_id,
                    "inner",
                )
                .groupBy("dst")
                .agg(F.sum(F.col("pagerank") * F.col("transition")).alias("rank"))
                .withColumnRenamed("dst", "sensor_id")
            )

            ranks = (
                vertices.join(
                    contributions,
                    "sensor_id",
                    "left",
                )
                .fillna({"rank": 0.0})
                .withColumn(
                    "pagerank",
                    (
                        F.lit(reset_probability)
                        + (F.lit(1 - reset_probability) * F.col("rank"))
                    ),
                )
                .select("sensor_id", "pagerank")
            )

        return ranks

    def build_features(
        self,
        graph_df: DataFrame,
    ) -> DataFrame:

        logger.info("Generating graph features")

        self.validate_input(graph_df)

        graph_df = graph_df.persist(StorageLevel.MEMORY_AND_DISK)

        try:
            degree = self.create_degree_features(graph_df)

            weights = self.create_weight_features(graph_df)

            neighbors = self.create_neighbor_features(graph_df)

            pagerank = self.create_pagerank_features(graph_df)

            result = (
                degree.join(
                    weights,
                    "sensor_id",
                    "left",
                )
                .join(
                    neighbors,
                    "sensor_id",
                    "left",
                )
                .join(
                    pagerank,
                    "sensor_id",
                    "left",
                )
                .fillna(
                    {
                        "out_degree": 0,
                        "in_degree": 0,
                        "degree": 0,
                        "out_weight_degree": 0.0,
                        "in_weight_degree": 0.0,
                        "weighted_degree": 0.0,
                        "neighbor_count": 0,
                        "avg_neighbor_weight": 0.0,
                        "pagerank": 0.0,
                    }
                )
            )

        finally:
            graph_df.unpersist()

        logger.info("Graph feature generation completed")

        return result
