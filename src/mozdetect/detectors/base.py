# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# The kind of data a detector is built for. A technique is written against the
# shape of its data -- per-build telemetry histograms, or the per-push runs of
# a CI test -- so the two are registered apart, and a caller asks for the set
# it can actually hand data to.
TELEMETRY_DETECTORS = "telemetry"
CI_DETECTORS = "ci"
DETECTOR_TYPES = (TELEMETRY_DETECTORS, CI_DETECTORS)

# Telemetry is what the detectors here were originally written for, and what
# alerting runs them on, so it's the set a caller gets when it doesn't say.
DEFAULT_DETECTOR_TYPE = TELEMETRY_DETECTORS


class UnknownDetectorTypeError(Exception):
    """Raised when a detector type that doesn't exist is used."""

    pass


def check_detector_type(detector_type):
    """Raises if `detector_type` isn't one of the types detectors are split into.

    Checked when detectors are registered, and when a set of them is
    requested, so a typo fails where it's made rather than quietly giving back
    an empty set of detectors.

    :param str detector_type: The detector type to check.

    :return str: The detector type.
    """
    if detector_type not in DETECTOR_TYPES:
        raise UnknownDetectorTypeError(
            f"Unknown detector type: {detector_type}. "
            f"Expecting one of: {', '.join(DETECTOR_TYPES)}"
        )
    return detector_type


class DetectorRegistry:
    _detectors = {detector_type: {} for detector_type in DETECTOR_TYPES}

    @staticmethod
    def add(detector_class, detector_name, detector_type=DEFAULT_DETECTOR_TYPE):
        """Add a detector to the registry of detectors.

        Detectors added to the registry will become available through the `get_detectors`
        method using the provided `detector_name`, for the given `detector_type`.

        :param str detector_name: Name of the detector.
        :param str detector_type: The kind of data the detector is for, either
            "telemetry" or "ci".
        """
        DetectorRegistry._detectors[check_detector_type(detector_type)][
            detector_name
        ] = detector_class

    @staticmethod
    def get_detectors(detector_type=DEFAULT_DETECTOR_TYPE):
        """Return all the detectors that were gathered for a kind of data.

        :param str detector_type: The kind of data to return the detectors of,
            either "telemetry" or "ci". Defaults to the telemetry detectors.
        """
        return DetectorRegistry._detectors[check_detector_type(detector_type)]


class BaseDetector:
    """Base class for all group detectors."""

    def __init_subclass__(cls, detector_name, detector_type=DEFAULT_DETECTOR_TYPE, **kwargs):
        super().__init_subclass__(**kwargs)
        DetectorRegistry.add(cls, detector_name, detector_type)

    def __init__(self, groups=None, **kwargs):
        """Initialize the detector.

        :param list groups: A list of DataFrame objects to compare between.
            Generally expected for there to be TWO groups to compare, but it's
            possible to have multiple to do a cross-comparison (assuming the
            detector supports this).
        """
        self.groups = groups

    def _coalesce_groups(self, groups):
        """Used to determine the groups to compare.

        The groups passed as an argument have a higher priority than the groups
        used to initialize the detector.

        :param list groups: A list of groups to compare or None.
        :return list: The groups that should be compared.
        """
        if not groups:
            if not self.groups:
                raise ValueError("Groups to compare have not been specified.")
            return self.groups
        return groups

    def detect_changes(self, groups=None, **kwargs):
        """Detect changes between two groups of data points.

        :param list groups: A list of the groups to compare.
        """
        pass
