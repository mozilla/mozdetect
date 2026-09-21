# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Runs the MWU change detection over a Perfherder series from production.

Find a signature to analyze:

    python examples/mwu_detection_run.py --project autoland --list --suite pdfpaint

Then detect the changes in it:

    python examples/mwu_detection_run.py --project autoland --signature <hash or id>
"""
import argparse

import mozdetect

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
        help="Signature hash (or signature ID) of the series to analyze",
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
        help=f"How many days of data to analyze (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--no-replicates",
        action="store_true",
        help="Compare the aggregated values only, not the replicates behind them",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        help="p-value at or below which two windows count as different",
    )
    parser.add_argument(
        "--cliffs-threshold",
        type=float,
        help="Smallest Cliff's delta worth calling a change",
    )
    parser.add_argument(
        "--back-window",
        type=int,
        help="How many pushes the base window covers",
    )
    parser.add_argument(
        "--fore-window",
        type=int,
        help="How many pushes the new window covers",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the signatures available instead of analyzing one",
    )
    parser.add_argument("--suite", help="Filter listed signatures by suite")
    parser.add_argument("--test", help="Filter listed signatures by test")
    parser.add_argument("--platform", help="Filter by platform, e.g. linux2404-64-shippable")

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


def detect_changes(args):
    series = get_signature_table(
        args.project,
        args.signature,
        framework=args.framework,
        days=args.days,
        replicates=not args.no_replicates,
        server=args.server,
    )

    timeseries = mozdetect.TreeherderTimeSeries(series)
    metadata = timeseries.metadata
    print(
        f"{metadata.get('name') or metadata['signature_hash']} on {metadata['platform']} "
        f"({metadata['repository_name']}, framework {metadata['framework_id']}): "
        f"{len(series)} pushes over {args.days} days"
    )

    settings = {
        "pvalue_threshold": args.pvalue_threshold,
        "cliffs_threshold": args.cliffs_threshold,
        "back_window": args.back_window,
        "fore_window": args.fore_window,
    }
    settings = {name: value for name, value in settings.items() if value is not None}

    detector = mozdetect.get_timeseries_detectors("ci")["mwu"](timeseries)
    detections = detector.detect_changes(**settings)

    if not detections:
        print("No changes detected.")
        return

    print(f"{len(detections)} change(s) detected:")
    print(detector.summarize(detections).to_string(index=False))


def main():
    args = parse_args()
    if args.list:
        list_signatures(args)
    else:
        detect_changes(args)


if __name__ == "__main__":
    main()
