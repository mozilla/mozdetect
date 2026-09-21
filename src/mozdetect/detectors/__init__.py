# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

from mozdetect.detectors.base import DEFAULT_DETECTOR_TYPE, DetectorRegistry

# Import all the classes from this submodule. This causes the subclass
# initialization that populates the DetectorRegistry.
package_dir = Path(__file__).resolve().parent
for _, module_name, _ in iter_modules([str(package_dir)]):
    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):
            # Add the class to this package's variables
            globals()[attribute_name] = attribute


def get_detectors(detector_type=DEFAULT_DETECTOR_TYPE):
    """Return the detectors that were gathered for a kind of data.

    :param str detector_type: The kind of data to return the detectors of,
        either "telemetry" or "ci". Defaults to the telemetry detectors.
    """
    return DetectorRegistry.get_detectors(detector_type)
