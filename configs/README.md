# Configs

Only the configs behind the final thesis results are kept. `README.md`
in the repository root maps each table and figure to its config.

- `final/`: RQ1 load, solar, and wind models, the ENTSO-E benchmark, the RQ1
  load and RQ2 evaluations, and the first-stage runs that feed the RQ2 price
  models (`*/price_inputs/`).
- `pricebase_sweep/`: RQ2 price models (P_base, P_gen, P_gen without raw weather).
- `rq3_clean/`: RQ3 cutoff grid (components 07:00 to 11:00, price models 07:00
  to 12:00 with and without reserve-market inputs).
- `rq3_cutoff_grid/`: EXAA-only benchmark at 12:00.
- `preprocessing/`: feature builders for the inputs of the configs above.
