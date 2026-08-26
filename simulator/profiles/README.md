# Checked-in analytic profiles

Use `python -m simulator profiles build ...` to create the normalized
msModeling analytic profile JSON consumed by the simulator.

The repository includes the 64K DeepSeek-V4-Flash grids used by the WebUI
examples, covering the documented 16, 24, 32, 40, and 48-die comparisons. Each
bundle records its exact msModeling commands, topology specs, source profile
composition, model/device metadata, and generation timestamp. `dsv4-smoke.json`
is a smaller grid for quick local checks.

Normalized JSON is checked in so a fresh clone can run the simulator without
installing msModeling. Raw Chrome traces remain ignored under `profiles/traces/`
because they are much larger. Regenerate a profile when changing the model,
device, msModeling revision, topology, or anchor grid, and review the metadata
diff together with the numeric points.
