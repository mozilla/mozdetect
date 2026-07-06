# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging
import numpy as np
import pandas
import base64
from io import BytesIO

from matplotlib import pyplot as plt

from datetime import timedelta
from prettytable import PrettyTable
from scipy.signal import find_peaks

from mozdetect.detectors import CDFSquaredDetector
from mozdetect.timeseries_detectors.base import BaseTimeSeriesDetector
from mozdetect.timeseries_detectors.detection import Detection
from mozdetect.utils import lowpass_filter

logger = logging.getLogger("CDFSquaredTimeSeries")


class CDFSquaredTimeSeriesDetector(BaseTimeSeriesDetector, timeseries_detector_name="cdf_squared"):
    """Analyzes a time series of histograms using CDFs to detect changes."""

    def __init__(self, timeseries, start_date=None, end_date=None, **kwargs):
        super().__init__(timeseries, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        # Set in detect_changes; the length of the multiday averaging window, reused
        # by the fast lane to locate each day's trailing baseline.
        self._multiday_average_days = 7

    def _coalesce_dates(self, start_date, end_date):
        return (
            self.start_date or start_date,
            self.end_date or end_date,
        )

    def _calculate_differences(self):
        """Calculates all the CDF differences across the full time series."""
        detector = CDFSquaredDetector()

        start_date, end_date = self._coalesce_dates(
            self.timeseries.cumulative_multiday_histograms["date"].min(),
            self.timeseries.cumulative_multiday_histograms["date"].max(),
        )

        current_date = start_date
        differences = pandas.DataFrame()
        while current_date <= end_date - timedelta(days=7):
            current_date_hist = self.timeseries.cumulative_multiday_histograms[
                self.timeseries.cumulative_multiday_histograms["date"] == current_date
            ][["bin", "cdf"]]
            next_date_hist = self.timeseries.cumulative_multiday_histograms[
                self.timeseries.cumulative_multiday_histograms["date"]
                == current_date + timedelta(days=7)
            ][["bin", "cdf"]]

            difference = detector.detect_changes(groups=[current_date_hist, next_date_hist])
            difference["date"] = [current_date + timedelta(days=6)]
            differences = pandas.concat(
                [differences, pandas.DataFrame(difference)], ignore_index=True
            )

            current_date += timedelta(days=1)

        logger.debug(differences)
        differences["filtered_sq_diff"] = lowpass_filter(differences["sq_diff"], 20, 100)

        return differences

    def _calculate_daily_differences(self):
        """Compares each day's single-day CDF against its trailing multiday baseline.

        Where `_calculate_differences` compares two `multiday_average_days` windows a
        week apart -- and therefore cannot react until a full post-change week has
        accumulated -- this compares the most recent single day against the trailing
        multiday average. That lets a large shift surface the day after it lands.

        :return pandas.DataFrame: The per-day differences with a `sq_diff` and `date`
            column, one row per day that has both a single-day histogram and a
            trailing baseline.
        """
        detector = CDFSquaredDetector()

        by_day = self.timeseries.cumulative_by_day_histograms
        multiday = self.timeseries.cumulative_multiday_histograms
        window = self._multiday_average_days

        # The multiday histogram labeled date D is the average of days [D, D+window-1],
        # so the trailing baseline for `current_date` (the average of the `window` days
        # ending the day before) is the multiday window starting `window` days earlier.
        start_date, end_date = self._coalesce_dates(by_day["date"].min(), by_day["date"].max())

        differences = pandas.DataFrame()
        current_date = start_date
        while current_date <= end_date:
            baseline_date = current_date - timedelta(days=window)
            baseline_hist = multiday[multiday["date"] == baseline_date][["bin", "cdf"]]
            today_hist = by_day[by_day["date"] == current_date][["bin", "cdf"]]

            if baseline_hist.empty or today_hist.empty:
                current_date += timedelta(days=1)
                continue

            difference = detector.detect_changes(groups=[baseline_hist, today_hist])
            difference["date"] = [current_date]
            differences = pandas.concat(
                [differences, pandas.DataFrame(difference)], ignore_index=True
            )
            current_date += timedelta(days=1)

        return differences

    def _find_fast_alerts(self, daily_differences, mad_k=6.0, min_days=14, require_confirmation=True):
        """Flags days whose single-day diff is well outside normal daily fluctuation.

        Uses a robust median + `mad_k` * MAD band to characterize "normal" daily
        movement. MAD is used rather than the standard deviation so that the handful
        of large regressions we are trying to catch don't inflate the band that
        defines normality. Only deviations large enough to clear the band fire, which
        keeps single-day sample noise from producing false positives.

        A single day can also shift the whole CDF hard and then snap back the next day
        -- a transient blip that is not a real regression, and which the band alone
        can't distinguish from a true change (the largest single-day spikes observed
        are often blips). When `require_confirmation` is set, a tripped day is only
        kept if the next calendar day also trips in the same direction, so blips that
        revert are dropped at the cost of one extra day of latency.

        :param pandas.DataFrame daily_differences: Output of `_calculate_daily_differences`.
        :param float mad_k: Number of (scaled) MADs beyond the median required to fire.
        :param int min_days: Minimum number of days needed to estimate a stable band;
            below this, no fast alerts are produced.
        :param bool require_confirmation: Require the next calendar day to trip in the
            same direction before flagging a day.
        :return pandas.DataFrame: Flagged days with `date`, `direction`, and
            `detection_type` columns.
        """
        empty = pandas.DataFrame(columns=["date", "direction", "detection_type"])
        if daily_differences.empty or len(daily_differences) < min_days:
            return empty

        metric = daily_differences["sq_diff"].astype(float)
        median = metric.median()
        mad = (metric - median).abs().median()

        if mad > 0:
            # 1.4826 scales the MAD to be consistent with the standard deviation for
            # normally distributed data, so mad_k is interpretable as "sigmas".
            band = mad_k * 1.4826 * mad
        else:
            std = metric.std(ddof=0)
            if std == 0:
                return empty
            band = mad_k * std

        upper = median + band
        lower = median - band

        tripped = daily_differences[(metric > upper) | (metric < lower)].copy()
        tripped["direction"] = tripped["sq_diff"].apply(lambda x: "up" if x > 0 else "down")

        if require_confirmation and not tripped.empty:
            # Keep a day only if the next calendar day tripped in the same direction.
            # A transient one-day blip reverts the next day and is therefore dropped.
            direction_by_date = dict(zip(tripped["date"], tripped["direction"]))
            confirmed = tripped.apply(
                lambda r: direction_by_date.get(r["date"] + timedelta(days=1)) == r["direction"],
                axis=1,
            )
            tripped = tripped[confirmed]

        if tripped.empty:
            return empty

        tripped["detection_type"] = "fast"
        return tripped[["date", "direction", "detection_type"]].reset_index(drop=True)

    def _collapse_consecutive(self, alerts, window_days):
        """Collapses same-direction alerts within `window_days` to their earliest date.

        A step change keeps firing the fast lane on consecutive days until the trailing
        baseline absorbs the new level, so a single regression produces a run of alerts.
        Keeping only the earliest preserves the fastest detection and drops the echoes.
        """
        if alerts.empty:
            return alerts

        alerts = alerts.sort_values("date").reset_index(drop=True)
        kept = []
        last_date_by_dir = {}
        for _, row in alerts.iterrows():
            direction = row["direction"]
            prev_date = last_date_by_dir.get(direction)
            if prev_date is not None and (row["date"] - prev_date).days <= window_days:
                continue
            kept.append(row)
            last_date_by_dir[direction] = row["date"]
        return pandas.DataFrame(kept, columns=alerts.columns).reset_index(drop=True)

    def _merge_and_dedup(self, smoothed, fast, dedup_window_days):
        """Merges smoothed and fast alerts, preferring the faster detection.

        Consecutive fast alerts are first collapsed to their earliest date. A smoothed
        alert is then dropped when a fast alert of the same direction already covers it
        within `dedup_window_days`, since the fast lane detected the same change sooner.

        :return pandas.DataFrame: The merged alerts sorted by date, with `date`,
            `direction`, and `detection_type` columns.
        """
        fast = self._collapse_consecutive(fast, dedup_window_days)

        if fast.empty:
            return smoothed.sort_values("date").reset_index(drop=True)

        kept_smoothed = []
        for _, s in smoothed.iterrows():
            covered = any(
                s["direction"] == f["direction"]
                and abs((f["date"] - s["date"]).days) <= dedup_window_days
                for _, f in fast.iterrows()
            )
            if not covered:
                kept_smoothed.append(s)

        kept_smoothed = pandas.DataFrame(kept_smoothed, columns=smoothed.columns)
        combined = pandas.concat([fast, kept_smoothed], ignore_index=True)
        return combined.sort_values("date").reset_index(drop=True)

    def _find_alerts(self, differences, alert_threshold=0.85):
        start_date, end_date = self._coalesce_dates(
            differences["date"].min(), differences["date"].max()
        )

        threshold = abs(differences["filtered_sq_diff"]).quantile(alert_threshold)

        peaks, _ = find_peaks(differences["filtered_sq_diff"], height=threshold, distance=3)
        valleys, _ = find_peaks(-differences["filtered_sq_diff"], height=threshold, distance=3)

        df_peaks = pandas.DataFrame({"date": differences.loc[peaks, "date"], "direction": "up"})
        df_valleys = pandas.DataFrame(
            {"date": differences.loc[valleys, "date"], "direction": "down"}
        )

        detections = (
            pandas.concat([df_peaks, df_valleys]).sort_values("date").reset_index(drop=True)
        )

        return detections

    def _get_interpolated_percentile(self, histogram, percentile):
        """Get an interpolated percentile value from the given histogram.

        :param pandas.DataFrame histogram: The histogram to get an interpolated
            percentile value from.
        :param float percentile:  The percentile to obtain a value for.
        """
        bucket_before = histogram[histogram["cdf"] <= percentile].max()
        bucket = histogram[histogram["cdf"] > percentile].min()
        bucket_after = histogram[histogram["cdf"] > percentile].iloc[1]

        x = np.array([bucket_before["cdf"], bucket["cdf"]])
        y = np.array([bucket["bin"], bucket_after["bin"]])
        return np.interp(percentile, x, y)

    def _describe_detection(self, detection):
        """Produce information about the detection."""
        if self.timeseries.cumulative_multiday_histograms is None:
            raise ValueError(
                "Cannot produce a description of the detection without a multiday average."
            )

        if detection.get("detection_type") == "fast":
            # The fast lane compared the single day against its trailing baseline, so
            # describe that same comparison: trailing multiday baseline vs. the day.
            before_histogram = self.timeseries.cumulative_multiday_histograms[
                self.timeseries.cumulative_multiday_histograms["date"]
                == detection["date"] - timedelta(days=self._multiday_average_days)
            ]
            after_histogram = self.timeseries.cumulative_by_day_histograms[
                self.timeseries.cumulative_by_day_histograms["date"] == detection["date"]
            ]
        else:
            before_histogram = self.timeseries.cumulative_multiday_histograms[
                self.timeseries.cumulative_multiday_histograms["date"]
                == detection["date"] - timedelta(days=7)
            ]
            after_histogram = self.timeseries.cumulative_multiday_histograms[
                self.timeseries.cumulative_multiday_histograms["date"] == detection["date"]
            ]

        merged_hist = pandas.merge(
            before_histogram, after_histogram, on="bin", suffixes=("_current", "_next")
        )
        merged_hist["diff"] = merged_hist["cdf_current"] - merged_hist["cdf_next"]
        total_diff = merged_hist["diff"].sum()

        detection_info = []
        detection_info.append(
            [
                "Total Samples",
                before_histogram["count"].sum(),
                after_histogram["count"].sum(),
            ]
        )
        detection_info.append(
            [
                "Interpolated Median",
                self._get_interpolated_percentile(before_histogram, 0.5),
                self._get_interpolated_percentile(after_histogram, 0.5),
            ]
        )
        detection_info.append(
            [
                "Interpolated p05",
                self._get_interpolated_percentile(before_histogram, 0.05),
                self._get_interpolated_percentile(after_histogram, 0.05),
            ]
        )
        detection_info.append(
            [
                "Interpolated p95",
                self._get_interpolated_percentile(before_histogram, 0.95),
                self._get_interpolated_percentile(after_histogram, 0.95),
            ]
        )
        detection_info.append(["CDF Diff", "", total_diff])
        detection_info.append(
            [
                "CDF Shift",
                "",
                "Left" if total_diff < -0.05 else "Right" if total_diff > 0.05 else "Mixed",
            ]
        )

        table = PrettyTable()
        table.field_names = ["Metric", "Before", "After"]
        table.add_rows(detection_info)

        logger.debug("Alert Generated: " + detection["date"].strftime("%Y-%m-%d"))
        logger.debug(table)

        detection_dict = {d[0]: d[1:] for d in detection_info}

        # Record which lane produced this detection ("fast" or "smoothed") so
        # downstream consumers can tell a next-day alert from a smoothed one.
        detection_dict["detection_type"] = detection.get("detection_type", "smoothed")

        # Store data representing the data before and after the detection
        detection_dict["additional_data"] = {
            "before": before_histogram[["bin", "cdf"]].dropna().to_dict(orient="list"),
            "after": after_histogram[["bin", "cdf"]].dropna().to_dict(orient="list"),
        }

        # Generate plot and add to detection info attachments
        detection_dict["attachments"] = []
        try:
            plot_image = self._plot_detection(
                detection["date"], before_histogram, after_histogram, detection["direction"]
            )
            detection_dict["attachments"].append(
                {
                    "data": plot_image,
                    "content_type": "image/png",
                    "file_name": f"cdf_difference_{detection['date']}.png",
                    "summary": (
                        "Graph showing the CDF difference between before and after the detection."
                    ),
                }
            )
        except Exception as e:
            logger.info(f"Failed to generate detection plot: {e}")

        return detection_dict

    def _plot_detection(self, detection_date, before_histogram, after_histogram, direction):
        """Create a graph showing the CDF before and after the detection.

        :param datetime detection_date: The date of the detection.
        :param DataFrame before_histogram: Histogram data before the detection.
        :param DataFrame after_histogram: Histogram data after the detection.
        :param str direction: The direction of the detection (up/down).
        :return str: Base64-encoded PNG image string.
        """
        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot both CDFs
        plt.plot(
            before_histogram["bin"],
            before_histogram["cdf"],
            label="Before",
            linewidth=2,
            color="blue",
        )
        plt.plot(
            after_histogram["bin"],
            after_histogram["cdf"],
            label="After",
            linewidth=2,
            color="red",
        )

        # Find the maximum bin value where CDF <= 0.99 for both histograms
        max_bin_before = before_histogram[before_histogram["cdf"] <= 0.99]["bin"].max()
        max_bin_after = after_histogram[after_histogram["cdf"] <= 0.99]["bin"].max()
        max_bin = max(max_bin_before, max_bin_after)
        if not pandas.isna(max_bin):
            plt.xlim(left=0, right=max_bin)

        # Add labels and title
        plt.xlabel("Bin", fontsize=12)
        plt.ylabel("Cumulative Distribution (CDF)", fontsize=12)
        plt.title(
            f"CDF Comparison: Detection on {detection_date.strftime('%Y-%m-%d')} "
            f"(Direction: {direction})",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Save to BytesIO buffer and encode as base64
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()
        return img_str

    def detect_changes(
        self,
        multiday_average_days=7,
        alert_threshold=0.85,
        fast_detection=True,
        fast_mad_k=6.0,
        fast_confirm=True,
        dedup_window_days=None,
        **kwargs,
    ):
        """Detects changes in a telemetry probe using the CDF squared detection method.

        Runs two lanes. The smoothed lane compares two `multiday_average_days` windows a
        week apart, maximizing sensitivity to gradual shifts while rejecting daily noise,
        but it cannot react until a full post-change week accumulates. The fast lane
        compares the most recent single day against its trailing baseline and fires only
        when the change is well outside normal daily fluctuation, letting large
        regressions be caught the day after they land. Fast detections are preferred over
        the smoothed detection of the same change during deduplication.

        :param int multiday_average_days: The number of days to use in the multiday average.
        :param float alert_threshold: Quantile threshold for the smoothed lane's peaks.
        :param bool fast_detection: Whether to run the fast (next-day) lane.
        :param float fast_mad_k: How many scaled MADs beyond the median a single-day diff
            must be for the fast lane to fire. Higher is more conservative.
        :param bool fast_confirm: Require the day after a fast trip to shift in the same
            direction before alerting. Suppresses transient one-day blips at the cost of
            one extra day of latency.
        :param int dedup_window_days: Window (in days) within which same-direction alerts
            are collapsed and a fast alert supersedes a smoothed one. Defaults to
            `multiday_average_days`.

        :return list: A list of Detection objects representing where a detection occurred.
        """
        self._multiday_average_days = multiday_average_days
        if dedup_window_days is None:
            dedup_window_days = multiday_average_days

        self.timeseries.get_cumulative_by_day(start_date=self.start_date, end_date=self.end_date)
        self.timeseries.get_multiday_average(days=multiday_average_days)

        differences = self._calculate_differences()
        pre_detections = self._find_alerts(differences, alert_threshold=alert_threshold)
        pre_detections = pre_detections.assign(detection_type="smoothed")

        if fast_detection:
            daily_differences = self._calculate_daily_differences()
            fast_detections = self._find_fast_alerts(
                daily_differences, mad_k=fast_mad_k, require_confirmation=fast_confirm
            )
            pre_detections = self._merge_and_dedup(
                pre_detections, fast_detections, dedup_window_days
            )

        detections = []
        for i in range(len(pre_detections)):
            detection = pre_detections.iloc[i]
            try:
                detection_info = self._describe_detection(detection)
            except Exception as e:
                logger.warning(f"Failed to describe detection on {detection['date']}: {e}")
                continue

            detections.append(
                Detection(
                    detection_info["Total Samples"][0],
                    detection_info["Total Samples"][1],
                    detection_info["CDF Diff"][1],
                    detection["date"],
                    detection["direction"],
                    **detection_info,
                )
            )

        return detections
