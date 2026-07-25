from typing import Optional

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


class SparkManager:
    def __init__(
        self,
        app_name: str = "TrafficAnalytics",
    ):
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None

    def create_spark_session(self):

        if self.spark is None:
            builder = (
                SparkSession.builder.appName(self.app_name)
                # ----------------------------
                # Memory configuration
                # ----------------------------
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "2g")
                .config("spark.sql.shuffle.partitions", "50")
                .config("spark.sql.files.maxPartitionBytes", "64MB")
                # ----------------------------
                # Delta
                # ----------------------------
                .config(
                    "spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension",
                )
                .config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                )
                # ----------------------------
                # Parquet compatibility
                # ----------------------------
                .config(
                    "spark.sql.parquet.int96RebaseModeInRead",
                    "LEGACY",
                )
                .config(
                    "spark.sql.parquet.int96RebaseModeInWrite",
                    "LEGACY",
                )
                .config(
                    "spark.sql.parquet.datetimeRebaseModeInRead",
                    "LEGACY",
                )
                .config(
                    "spark.sql.parquet.datetimeRebaseModeInWrite",
                    "LEGACY",
                )
            )

            self.spark = configure_spark_with_delta_pip(builder).getOrCreate()

            self.spark.sparkContext.setLogLevel("WARN")

        return self.spark

    def stop(self):

        if self.spark:
            self.spark.stop()

            self.spark = None
