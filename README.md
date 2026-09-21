# mozdetect
A python package containing change point detection techniques for use at Mozilla.

# Setup, and Development

## Setup

Install `uv` first using the following:

```
python -m pip install uv
```

Install `poetry` using the following:

```
python -m pip install poetry
```

## Running

Next, run the following to build the package, and install dependencies. This step can be skipped though since `uv run` will implicitly build the package:

```
uv sync
```

Run a script that uses the built module with the following:

```
uv run my_script.py
```

## Testing Change Detection Techniques

This section provides an overview about how to add and test new or existing change detection techniques.

### Adding New Techniques

All techniques are defined in two parts. The first part is the detector itself that compares two (or more) groups to each other, and returns the result of that comparison. The second part is a timeseries detector that runs across a full timeseries (e.g. a TelemetryTimeSeries) and uses the detector from the first part to detect changes.

Both of these should be defined for any new techniques. This makes it possible to make different timeseries detectors using the same underlying detector technique. See `src/mozdetect/detectors/cdf_squared.py` for an example implementation of a detector, and `src/mozdetect/detectors/cdf_squared.py` for an example implementation of the timeseries detector. Note that the detectors will need to be subclasses of the `BaseDetector`, and `BaseTimeSeriesDetector`, respectively. Furthermore, they need to specify a name that will be used to access them, e.g. `cdf_squared`, through the `detector_name`, and `timeseries_detector_name` class initialization arguments. These names will be used to access the detectors from the return value of `get_detectors`/`get_timeseries_detectors`.

Detectors also specify the kind of data they're for through the `detector_type` class initialization argument, since a technique is written against the shape of its data: `telemetry` for the per-build histograms of a probe, and `ci` for the per-push runs of a CI test. It defaults to `telemetry`, which is also the set `get_detectors`/`get_timeseries_detectors` return when they're not asked for one, so a caller has to ask for the other kind explicitly:
```python
mozdetect.get_timeseries_detectors()  # the telemetry techniques
mozdetect.get_timeseries_detectors("ci")  # the CI ones
```
The two sets are kept apart rather than merged, so a name means one technique for one kind of data, and asking for a type that doesn't exist raises `UnknownDetectorTypeError` rather than giving back an empty set.

The `TelemetryTimeSeries` object provides an interface for accessing the data with some helper methods. However, if those are not enough, it's possible to access the raw data that the time series object was built with through `TelemetryTimeSeries.raw_data`. The same is true for `TreeherderTimeSeries` objects.

The detector only needs to return a dictionary with information about the comparison. However, the timeseries detector must return a list of `Detection` objects that contain information about the changes detected.

### Testing Techniques

After the new technique was added, create a new testing script. This script can exist anywhere, but there's a special folder that can be added to the top-level of the repo called `sample-scripts` that can contain the script and it will be ignored when making commits. See the example in `examples/sample_detection_run.py` for how to run the detection. It can be run using the following from the top-level of the repo:
```
uv run examples/sample_detection_run.py
```

At the moment, the only detection techniques available use data from BigQuery. This means that you will need to login locally, and ensure that you have access to the `mozdata` project. Follow [these instructions](https://cloud.google.com/sdk/docs/install) for how to install the tool, then run the following to login and set the project:
```
gcloud auth login --update-adc
gcloud config set project mozdata
```

The key things to do in the script are calling `get_metric_table` to get the data, creating a `TelemetryTimeSeries` with the data, and then calling the change detection technique with the timeseries object as an argument. The change detection technique class is obtained from `mozdetect.get_timeseries_detectors()["name-of-detector"]` (name of the detector is given when creating the detector class). Calling `detect_changes()` on the resulting object will trigger the change detection, and return a list of `Detection` objects that describe the change that was detected.

### Perfherder Data

Timeseries data from Perfherder can be pulled from a Treeherder API server with `mozdetect.treeherder_query`, which works the same way as `telemetry_query` does for BigQuery, minus the login: the data is public and nothing is written, so it can be pointed at production from anywhere.

Use `get_signatures` to help with finding series with specific characteristics, and `get_signature_table` to get the data for that series. The table holds one row per data point, with the measurements behind each point in a `trials` column - the replicates when the jobs reported any, and the aggregated value standing in for them otherwise. The signature's metadata (e.g. `lower_is_better`, `alert_threshold`, etc.) are found on the frame's `attrs`. Since change detection usually works over pushes rather than jobs, and a push can hold several data points for the same signature (retriggers or backfills), the data is aggregated to have one row per push with all retriggers combined in them. Pass `per_push=False` to disable that behaviour.

Both of those query the API. Code running inside Treeherder itself has the same data in the database, so `get_signature_table_from_query` builds the same table out of a Django query's rows instead, letting a technique developed here against production data run unchanged in alerting:
```python
from mozdetect import (
    QUERY_FIELDS,
    TreeherderTimeSeries,
    get_signature_table_from_query,
)

datums = PerformanceDatum.objects.filter(
    signature=signature, push_timestamp__gte=since
).values(*QUERY_FIELDS)

signature_fields = PerformanceSignature.objects.filter(id=signature.id).values().first()

table = get_signature_table_from_query(datums, signature_fields)
```
Both arguments are the rows of a `values()` query. `QUERY_FIELDS` always selects the replicates, which joins them in the way Perfherder's own endpoint reads them: one row per replicate, with the data point's fields repeated across them, and a null replicate value for the jobs that didn't report any (those fall back to the aggregated value). Everything the signature's own query selected becomes the metadata on `attrs`, under the names it selected them with, so a bare `values()` gathers all the signature settings. Note that the related rows come back as the ids they are, so a query wanting the platform or repository by name has to ask for `platform__platform` or `repository__name`. Either table can then be handed to `TreeherderTimeSeries`.

See `examples/treeherder_query_run.py` for a script that does both, which can be run using the following from the top-level of the repo:
```
uv run examples/treeherder_query_run.py --project autoland --list --suite pdfpaint
uv run examples/treeherder_query_run.py --project autoland --signature <hash or id>
```

### Using New Techniques in Alerting/Monitoring

Once a new technique is added, a new release of mozdetect will need to be produced. From there, an update in Treeherder will be needed for the mozdetect package along with a new deployment. Once deployed, it will be usable.

Currently, mozdetect is only used for alerting on telemetry probes so in the `monitor` field that is added to the probe(s), the field `change_detection_technique` will need to be used to specify the name of the change detection technique that was added with the detector class - only the timeseries detector classes are used in alerting, and monitoring. Additional arguments to the technique can also be provided through the `change_detection_args` field.


## Pre-commit checks

Pre-commit linting checks must be setup like this (run within the top-level of this repo directory):

```
uv sync
uv run pre-commit install
```

## Running tests, and linting

Tests all reside in the `tests/` folder and can be run using:

```
uv run pytest
```

Linting is performed through pre-commit when you commit, however, it's possible to run it directly without performing a commit:
```
uv run pre-commit run --all-files
```
