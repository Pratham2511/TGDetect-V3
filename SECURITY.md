# Security policy

## Supported version

Security fixes are applied to the current `main` branch. Historical snapshots and separate legacy model artifacts are not maintained by this repository.

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Pratham2511/TGDetect-V3/security/advisories/new). Include the affected component, relevant version or commit, reproduction steps, likely impact, and a minimal sanitized example. Do not open a public issue with exploitable details before the maintainer has reviewed the report.

Never attach credentials, private telemetry, or identifiable user data. If sensitive information has already been exposed, remove it from shared artifacts and rotate affected credentials.

## Local operation

The documented configuration runs the API and frontend locally. Uploaded logs are stored on the machine under ignored `backend/dashboard/data/graph_uploads/`. Review data retention and access before using real organizational telemetry.

Public deployment requires authentication, authorization, TLS, request limits, and deployment-specific retention controls. Model scores and graph paths are analyst evidence, not an automatic security decision or a guarantee that activity is safe.

The downloader verifies the archive and individual file checksums. Replay verifies the original checkpoint and frozen source. Do not bypass those integrity checks to resolve an installation error.
