import sys
import logging
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array


# Allow importing project modules
sys.path.append(str(Path(__file__).resolve().parent.parent))


from spark_manager import SparkManager
from graph_features import TrafficGraphFeatureBuilder
from clean_data import TrafficFeatureBuilder


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SILVER_GRAPH_PATH = "hdfs://localhost:9000/silver/graph"

SILVER_TRAFFIC_PATH = "hdfs://localhost:9000/silver/traffic"

GOLD_PATH = "hdfs://localhost:9000/gold/gnn_features"


# ---------------------------------------------------------
# Convert Wide Traffic -> Long
# ---------------------------------------------------------


def convert_traffic_to_long(df):

    if "sensor_id" in df.columns and "speed" in df.columns:
        return df

    sensor_columns = [c for c in df.columns if c != "timestamp"]

    stack_expression = ", ".join(
        [f"'{sensor}', `{sensor}`" for sensor in sensor_columns]
    )

    return df.selectExpr(
        "timestamp",
        f"""
            stack(
                {len(sensor_columns)},
                {stack_expression}
            )
            as (sensor_id, speed)
            """,
    ).withColumn("speed", F.col("speed").cast("double"))


# ---------------------------------------------------------
# Convert Spark ML Vector columns
# ---------------------------------------------------------


def convert_vector_columns(df):

    vector_columns = []

    for field in df.schema.fields:
        if "VectorUDT" in str(field.dataType):
            vector_columns.append(field.name)

    for c in vector_columns:
        logger.info(f"Converting vector column: {c}")

        df = df.withColumn(c, vector_to_array(F.col(c)))

    return df


# ---------------------------------------------------------
# Remove large unused columns
# ---------------------------------------------------------


def remove_large_columns(df):

    remove = ["features", "scaled_features", "numeric_features"]

    for c in remove:
        if c in df.columns:
            logger.info(f"Dropping large column: {c}")

            df = df.drop(c)

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    spark_manager = SparkManager(app_name="traffic-gold-layer")

    spark = spark_manager.create_spark_session()

    try:
        # =================================================
        # Load Traffic
        # =================================================

        logger.info("Loading traffic data")

        traffic_df = spark.read.format("delta").load(SILVER_TRAFFIC_PATH)

        traffic_df = convert_traffic_to_long(traffic_df)

        traffic_df.printSchema()

        # =================================================
        # Load Graph
        # =================================================

        logger.info("Loading graph data")

        graph_df = spark.read.format("delta").load(SILVER_GRAPH_PATH)

        graph_df.printSchema()

        # =================================================
        # Graph Features
        # =================================================

        logger.info("Building graph features")

        graph_builder = TrafficGraphFeatureBuilder()

        graph_features = graph_builder.build_features(graph_df)

        graph_features.show(5)

        # =================================================
        # Traffic Features
        # =================================================

        logger.info("Building traffic features")

        traffic_builder = TrafficFeatureBuilder()

        traffic_features = traffic_builder.build_features(traffic_df)

        traffic_features.show(5)

        # =================================================
        # Join
        # =================================================

        logger.info("Joining features")

        gold_features = traffic_features.join(
            graph_features, on="sensor_id", how="left"
        )

        graph_numeric_cols = [
            "out_degree",
            "in_degree",
            "degree",
            "out_weight_degree",
            "in_weight_degree",
            "weighted_degree",
            "neighbor_count",
            "avg_neighbor_weight",
            "pagerank",
        ]

        existing_cols = [c for c in graph_numeric_cols if c in gold_features.columns]

        gold_features = gold_features.fillna(0, subset=existing_cols)

        logger.info("Before vector conversion")

        gold_features.printSchema()

        # =================================================
        # Prepare for Delta
        # =================================================

        logger.info("Converting vector columns")

        gold_features = convert_vector_columns(gold_features)

        gold_features = remove_large_columns(gold_features)

        logger.info("Final schema")

        gold_features.printSchema()

        # Reduce write pressure

        gold_features = gold_features.repartition(50)

        # =================================================
        # Write Gold
        # =================================================

        logger.info("Writing gold layer")

        (
            gold_features.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(GOLD_PATH)
        )

        logger.info("Gold layer created successfully")

    finally:
        spark_manager.stop()


if __name__ == "__main__":
    main()
