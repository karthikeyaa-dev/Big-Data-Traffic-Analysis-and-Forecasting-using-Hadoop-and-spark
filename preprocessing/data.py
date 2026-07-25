from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = (
    SparkSession.builder.appName("ReadGoldTraffic")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Delta table path
path = "hdfs://localhost:9000/silver/graph"

# Read Delta table
df = spark.read.format("delta").load(path)

# Show information
print("Columns:")
for col in df.columns:
    print(col)

print("\nNumber of columns:", len(df.columns))
print("\nNumber of rows:", df.count())

print("\nSample data:")
df.show(5, truncate=False)

# Save as Parquet
output_path = "file:///home/karthikeya/Development/Machine_Learning/Traffic_Analytics/graph_parquet"

(df.write.mode("overwrite").parquet(output_path))

print(f"\nParquet saved to: {output_path}")

spark.stop()
