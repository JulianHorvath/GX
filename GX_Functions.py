import great_expectations as gx
from datetime import datetime
import pytz
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, DoubleType, TimestampType


class GXRunner:
    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.context = gx.get_context()
        self._custom_expectations = {}
        self._custom_actions = {}

    def load_spark_df(self, catalog: str, schema: str, table: str):
        return self.spark.read.table(f"{catalog}.{schema}.{table}")

    # ------------------------
    # Dynamic Expectations generator
    # ------------------------

    def register_expectation(self, name: str, func):
        self._custom_expectations[name] = func

    def get_expectation(self, name: str):
        if name in self._custom_expectations:
            return self._custom_expectations[name]
        raise ValueError(f"Expectation '{name}' is not registered.")

    # ------------------------
    # Actions generator
    # ------------------------

    def register_action(self, name: str, action_class, default_params=None):
        self._custom_actions[name] = {"class": action_class, "params": default_params or {}}

    def build_action(self, name: str, override_params=None):
        if name not in self._custom_actions:
            raise ValueError(f"Action '{name}' is not registered.")
        action_def = self._custom_actions[name]
        action_class = action_def["class"]
        params = {**action_def["params"], **(override_params or {})}
        return action_class(**params)

    # ------------------------
    # Main validation runner
    # ------------------------

    def run_tests(self,
                  df,
                  expectations_config: list,
                  action_list,
                  data_src_name: str,
                  data_asset_name: str,
                  batch_definition_name: str,
                  suite_name: str,
                  validation_definition_name: str,
                  checkpoint_name: str):

        # Datasource and batch
        data_source = self.context.data_sources.add_spark(data_src_name)
        data_asset = data_source.add_dataframe_asset(name=data_asset_name)
        batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
        batch_parameters = {"dataframe": df}
        batch = batch_definition.get_batch(batch_parameters=batch_parameters)

        # Suite definition
        suite = gx.ExpectationSuite(name=suite_name)
        self.context.suites.add(suite)

        for item in expectations_config:
            func = item["func"]
            params = item.get("params", {})
            expectation = func(**params)
            suite.add_expectation(expectation)

        # Validation definition
        validation_definition = gx.ValidationDefinition(data=batch_definition, suite=suite, name=validation_definition_name)
        self.context.validation_definitions.add(validation_definition)

        # Checkpoint w/Action
        checkpoint = gx.Checkpoint(
            name=checkpoint_name,
            validation_definitions=[self.context.validation_definitions.get(validation_definition_name)],
            actions=action_list,
            result_format={"result_format": "COMPLETE"}
        )

        self.context.checkpoints.add(checkpoint)
        checkpoint_results = checkpoint.run(batch_parameters=batch_parameters)

        # Return results as PySpark DF
        for _, validation_result in checkpoint_results.run_results.items():
            checkpoint_results = validation_result.to_json_dict()

        suite_name = checkpoint_results.get("suite_name")
        validation_time = checkpoint_results.get("meta", {}).get("validation_time")
        if validation_time:
            validation_time = datetime.fromisoformat(validation_time.replace("Z", "+00:00"))
        else:
            validation_time = None

        rows = []
        for result in checkpoint_results["results"]:
            row = Row(
                result.get("success"),
                result["expectation_config"].get("type"),
                result["expectation_config"]["kwargs"].get("column"),
                result["result"].get("element_count"),
                result["result"].get("unexpected_count"),
                result["result"].get("unexpected_percent"),
                suite_name,
                validation_time
            )
            rows.append(row)

        schema = StructType([
            StructField("success", BooleanType(), True),
            StructField("type", StringType(), True),
            StructField("column", StringType(), True),
            StructField("element_count", IntegerType(), True),
            StructField("unexpected_count", IntegerType(), True),
            StructField("unexpected_percent", DoubleType(), True),
            StructField("suite_name", StringType(), True),
            StructField("validation_time", TimestampType(), True)
        ])

        return self.spark.createDataFrame(rows, schema=schema)