# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Pulls a Perfherder timeseries from Treeherder production and prints it.

Find a signature to analyze:

    python examples/treeherder_query_run.py --project autoland --list --suite pdfpaint

Then read its series:

    python examples/treeherder_query_run.py --project autoland --signature <hash or id>
"""
import argparse

from mozdetect.treeherder_query import (
    DEFAULT_DAYS,
    DEFAULT_SERVER,
    get_signature_table,
    get_signatures,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        required=True,
        help="Repository the series belongs to, e.g. autoland",
    )
    parser.add_argument(
        "--signature",
        help="Signature hash (or signature ID) of the series to read",
    )
    parser.add_argument(
        "--framework",
        type=int,
        help="Framework ID, to pick between signatures sharing a hash",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Treeherder server to read from (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"How many days of data to read (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--from-date",
        help="Read from this date (YYYY-MM-DD) instead of the last --days days",
    )
    parser.add_argument(
        "--to-date",
        help="Read up to this date (YYYY-MM-DD). Defaults to now",
    )
    parser.add_argument(
        "--no-replicates",
        action="store_true",
        help="Read the aggregated values only, not the replicates behind them",
    )
    parser.add_argument(
        "--no-retriggers",
        action="store_true",
        help="Drop retriggered jobs from the series",
    )
    parser.add_argument(
        "--no-per-push",
        action="store_true",
        help="Print one row per data point rather than one row per push",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the signatures available instead of reading a series",
    )
    parser.add_argument("--suite", help="Filter listed signatures by suite")
    parser.add_argument("--test", help="Filter listed signatures by test")
    parser.add_argument("--platform", help="Filter by platform, e.g. linux2404-64-shippable")
    parser.add_argument(
        "--no-subtests",
        action="store_true",
        help="Only list summary signatures",
    )

    args = parser.parse_args()
    if not args.list and not args.signature:
        parser.error("either --signature or --list is required")
    return args


def list_signatures(args):
    signatures = get_signatures(
        args.project,
        framework=args.framework,
        suite=args.suite,
        test=args.test,
        platform=args.platform,
        subtests=not args.no_subtests,
        days=args.days,
        server=args.server,
    )

    print(f"{len(signatures)} signature(s) on {args.project}:")
    for signature in signatures:
        test = signature.get("test")
        print(
            f"  {signature['signature_hash']}  {signature['suite']}"
            f"{' ' + test if test else ''} "
            f"({signature['machine_platform']}, framework {signature['framework_id']})"
        )


def read_series(args):
    series = get_signature_table(
        args.project,
        args.signature,
        framework=args.framework,
        days=args.days,
        replicates=not args.no_replicates,
        server=args.server,
        no_retriggers=args.no_retriggers,
        from_date=args.from_date,
        to_date=args.to_date,
        per_push=not args.no_per_push,
    )
    metadata = series.attrs

    print(
        f"{metadata.get('name') or metadata['signature_hash']} on {metadata['platform']} "
        f"({metadata['repository_name']}, framework {metadata['framework_id']}, "
        f"signature {metadata['signature_id']})"
    )
    print(
        f"lower_is_better={metadata['lower_is_better']}, "
        f"alert_threshold={metadata.get('alert_threshold')}, "
        f"unit={metadata.get('measurement_unit') or 'n/a'}"
    )

    if not args.no_per_push:
        print(
            f"{len(series)} pushes, {int(series['value_count'].sum())} data points, "
            f"{int(series['trial_count'].sum())} trials"
            f"{' (replicates off)' if args.no_replicates else ''}"
        )
        print(
            series[
                ["push_timestamp", "push_id", "revision", "value", "value_count", "trial_count"]
            ].to_string(index=False)
        )
    else:
        trials = (
            len(series)
            if args.no_replicates
            else sum(len(replicates) for replicates in series["replicates"])
        )
        print(
            f"{series['push_id'].nunique()} pushes, {len(series)} data points, {trials} trials"
            f"{' (replicates off)' if args.no_replicates else ''}"
        )
        columns = ["push_timestamp", "push_id", "revision", "job_id", "value"]
        print(series[columns].to_string(index=False))


def main():
    args = parse_args()
    if args.list:
        list_signatures(args)
    else:
        read_series(args)


if __name__ == "__main__":
    main()
