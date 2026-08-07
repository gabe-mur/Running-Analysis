# Local data and repository privacy

The application is designed to keep athlete data local. The repository tracks
code, tests, documentation, and sanitized example configuration only.

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

Before publishing a branch, run `git status --short --ignored` and verify that
only intended source files are staged. Do not force-add an ignored activity,
database, local configuration, or generated report.
