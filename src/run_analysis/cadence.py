"""The single conversion from recorded cadence to total steps per minute.

Recorded cadence is not one quantity. Garmin's ``RunCadence`` extension
reports strides per minute for one leg, while a plain TCX ``Cadence`` element
already reports total steps per minute. Doubling one but not the other in more
than one place is how a run ends up classified as a walk in one analysis and a
run in another.

Every consumer must therefore read cadence through :func:`cadence_spm` (or
``Trackpoint.cadence_spm``, which delegates here). The raw ``cadence`` value
and its ``cadence_source`` stay on the trackpoint and in the database so the
conversion can always be audited back to what the device actually wrote.
"""

from __future__ import annotations


#: ``cadence_source`` value set by the TCX parser for Garmin's one-sided
#: running-cadence extension.
ONE_SIDED_CADENCE_SOURCE = "run_cadence_extension"

#: FIT ``record.cadence`` for running is one-sided too: strides per minute for
#: a single leg, with ``fractional_cadence`` carrying the half-stride. It gets
#: its own name so a stored value's provenance stays readable, but it must
#: double exactly the same way. Omitting it would not raise anything -- it
#: would halve every FIT cadence, and :mod:`classification` would then file
#: each of those runs as a walk and drop them from the model.
FIT_ONE_SIDED_CADENCE_SOURCE = "fit_record_cadence"

#: Sources whose recorded value must be doubled to become total steps/minute.
_ONE_SIDED_SOURCES = frozenset({ONE_SIDED_CADENCE_SOURCE, FIT_ONE_SIDED_CADENCE_SOURCE})


def cadence_spm(recorded: float | int | None, source: str | None) -> float | None:
    """Total steps per minute, or ``None`` when there is no usable reading.

    A recorded zero means the device reported no turnover for that sample; it
    is absence of evidence rather than a cadence of zero, so it is dropped
    instead of dragging averages down.
    """

    if recorded is None:
        return None
    value = float(recorded)
    if value <= 0:
        return None
    if source in _ONE_SIDED_SOURCES:
        value *= 2.0
    return value


def is_one_sided_source(source: str | None) -> bool:
    """Whether ``source`` records a single leg rather than total steps."""
    return source in _ONE_SIDED_SOURCES
