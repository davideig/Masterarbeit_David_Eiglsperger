# Data Directory

Heavy data is not part of the clean release repo.

Expected local layout:

```text
data/
  processed/
  cache/
  raw/
```

Use one of the following approaches:

1. Download a prepared thesis feature pack and unpack it into `data/`.
2. Rebuild the processed files with configs under `configs/preprocessing/`.
3. Use sample data under `data/sample/` if provided by a release package.

See `docs/data_catalog.md` for the required files per model.
