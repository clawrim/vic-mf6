# VIC-MF6

VIC-MF6 is a modeling framework for statewide, coupled surface–subsurface hydrologic simulation in New Mexico, integrating the [VIC](https://github.com/UW-Hydro/VIC) and [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6) models. It couples a downscaled version of [Yang et al. (2019)](https://doi.org/10.1029/2018WR024178) and [a MODFLOW 6 converted version](https://github.com/clawrim/rgtihm) of [the Rio Grande Transboundary Integrated Hydrologic Model (RGTIHM)](https://doi.org/10.3133/sir20195120). This framework requires [modifications](https://github.com/clawrim/modflow6/tree/vic-mf6) to handling of geologic layers in MODFLOW 6. We used [the VIC Classic to Image converter](https://github.com/clawrim/vic_classic_to_image) to convert the New Mexico VIC Class model to an Image model for time-over-space coupling.

This figure shows the flowchart of the VIC-MF6 coupling framework:
<img width="879" height="1247" alt="image" src="https://github.com/user-attachments/assets/7c6150ce-a47d-43f3-b7b7-874adb94559e" />
