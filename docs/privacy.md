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

Optional Intervals.icu sync sends your API credentials to Intervals.icu over
verified HTTPS and downloads activity files and available wellness data. The
credentials are stored in the private local data folder, not the application
bundle or Git repository. Disconnecting does not delete already imported runs.

Explicit terrain downloads request broad one-degree Tilezen map tiles from AWS
S3 and the provider's attribution document. They do not send activity files,
heart rate or routes. Normal elevation analysis reads local terrain and bridge
geometry and does not make elevation requests. Downloaded maps are not included
in the shared app ZIP.

The packaged Mac app keeps these files under
`~/Library/Application Support/Running Coach/`, outside the application bundle.
Replacing the app preserves this folder. The separate clean demo uses
`~/Library/Application Support/Running Coach Clean/`. Source-mode launches keep
their data in the project folder instead. No personal data is bundled in either
application.

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
