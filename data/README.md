# Data

Only the cluster assignments used by the final thesis models are included.
Each file maps the ICON-D2 grid points to spatial clusters.

| File | Used for |
|---|---|
| `clustering/icon_d2_clustering_c25.parquet` | 25 clusters of the OBTF load model (population-weighted weather) |
| `clustering/icon_d2_mastr_solar_tso_c25.csv` | 25 capacity-weighted solar clusters per TSO proxy area |
| `clustering/icon_d2_mastr_wind_c100.csv` | 100 capacity-weighted wind clusters including offshore |
| `clustering/icon_d2_clustering_c{1,2,5,8,12,16,40,64,100}.parquet` | Weather clusters of the price model (two clusters in the final model, the others for the cluster sweep in Appendix A3.1) |

The model input files are a separate download (`thesis_model_inputs.zip`, see
"Rerunning the models" in the main README). Unpacked in the repository root,
they go to `data/processed/`.
