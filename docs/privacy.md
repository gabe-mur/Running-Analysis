# Privacy

Running Coach stores your activity history on this computer. It has no account,
cloud database, analytics tracker, advertising code, or third-party frontend
assets.

## What stays local

The following paths are intentionally ignored by Git:

- `TCX/` and `uploads/`: raw activity exports and newly uploaded activities;
- `data/`: SQLite history, weather responses, location-derived cache entries,
  and the local privacy salt;
- `output/`: generated reports, model results, and diagnostics;
- `config.yaml` and `config.local.yaml`: personal physiology, preferences, and
  any explicit location mappings;
- `run_overrides.csv`: activity IDs, health tags, perceived exertion, and notes;
- `docs/local/`: athlete-specific audits and validation results.

Open-Meteo requests are optional and disabled in the example configuration.
Historical weather and planned forecasts have separate Settings opt-ins. When
enabled, the application sends required dates or planned timestamps and a route
centroid rounded and privacy-jittered according to the local configuration. Raw
routes are not sent by the weather integration.

Entering a ZIP code for a no-GPS run sends that ZIP to Open-Meteo's geocoding
service once when you save it. The resulting approximate location is stored
locally; subsequent weather requests use the rounded and randomized location.

The normal launcher listens only on `127.0.0.1`, so other devices cannot open
the app. The server accepts only local hostnames, blocks framing and browser
permissions, sends no-referrer and no-cache headers, and limits scripts,
styles, images, and network connections to the local app itself.

On macOS and Linux, files and folders created by the app for settings, uploads,
databases, caches, reports, and overrides are restricted to the current OS
user. These permissions are an extra local safeguard, not encryption.

## What to remember

- Raw GPS points, heart rate, health tags, notes, ZIP-derived locations, and
  profile settings are sensitive even though they remain local.
- Computer backups, cloud-synced folders, malware, or another process running
  as your OS user can still copy local files. Put the repository in a
  non-synced folder if that matters for your threat model.
- `.gitignore` prevents ordinary Git staging; it does not erase previously
  committed data or stop a deliberate force-add.

Before publishing a branch, run `git status --short --ignored` and verify that
only intended source files are staged. Do not force-add an ignored activity,
database, local configuration, or generated report.
