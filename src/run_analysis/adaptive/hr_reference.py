"""Current athlete settings are not the unit scale of an adopted load ledger.

Reopen only a verified, already pinned reference. Never rebase seed history,
issued forecasts or completions when the athlete changes HR settings. Generic
sandbox/research resume retains its strict configuration check.
"""
import json
from pathlib import Path
import sqlite3


def frozen_live_config(config, *, root=None, connection=None):
    if root is None:
        row = connection.execute("SELECT value_json FROM app_state WHERE key='adaptive_runtime'").fetchone()
        if row is None:
            raise ValueError('No verified adopted HR accounting reference; no settings were saved')
        pointer = json.loads(row[0])
        if pointer.get('engine') != 'adaptive_frequency_v3':
            raise ValueError('Adopted HR accounting engine mismatch; no settings were saved')
        root = pointer['root']
    root = Path(root)
    marker = json.loads((root/'execution-sandbox.json').read_text())
    from run_analysis.continuous_training_dose import ContinuousDoseReference
    if (marker.get('version') != 1
            or marker.get('accounting_key') != ContinuousDoseReference('sandbox upload accounting').accounting_key):
        raise ValueError('Adopted load-accounting identity changed; no settings were saved')
    reference = json.loads(marker['measurement_context'])
    if (set(reference) != {'max_hr', 'resting_hr'}
            or any(type(v) is not int for v in reference.values())
            or not 0 < reference['resting_hr'] < reference['max_hr'] <= 250):
        raise ValueError('Invalid adopted HR accounting reference; no settings were saved')
    with sqlite3.connect((root/'sandbox.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
        row = db.execute("SELECT value_json FROM adaptive_bridge_state WHERE key='normalization'").fetchone()
    expected = dict(reference, timezone_default=marker['timezone'])
    if row is None or json.loads(row[0]) != expected:
        raise ValueError('Adopted HR accounting records disagree; no settings were saved')
    if (config['timezone_default'] != marker['timezone']
            or config['moving_time'] != marker['moving_time']):
        raise ValueError('Timezone/movement accounting changed; explicit migration required. No settings were saved.')
    # Current watch/profile maxima and zone edges are athlete settings, not
    # a request to redefine the adopted load unit. They can move in either
    # direction. Actual traces and requested effort bands still enforce the
    # pinned normalization domain at their consumption boundaries; nothing
    # here clips HR, expands that domain, or rewrites a saved measurement.
    # Current zones/menu/profile are prospective inputs; only the numerical
    # load unit remains pinned. Sandbox opening independently checks identity.
    return dict(config, **reference)
