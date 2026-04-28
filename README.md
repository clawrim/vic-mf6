# VIC-MF6

VIC-MF6 is a modeling framework for statewide, coupled surface–subsurface hydrologic simulation in New Mexico, integrating the [VIC](https://github.com/UW-Hydro/VIC) and [MODFLOW 6](https://github.com/MODFLOW-ORG/modflow6) models. It couples a downscaled version of [Yang et al. (2019)](https://doi.org/10.1029/2018WR024178) and [a MODFLOW 6 converted version](https://github.com/clawrim/rgtihm) of [the Rio Grande Transboundary Integrated Hydrologic Model (RGTIHM)](https://doi.org/10.3133/sir20195120). This framework requires [modifications](https://github.com/clawrim/modflow6/tree/vic-mf6) to handling of geologic layers in MODFLOW 6. We used [the VIC Classic to Image converter](https://github.com/clawrim/vic_classic_to_image) to convert the New Mexico VIC Class model to an Image model for time-over-space coupling.

This figure shows the flowchart of the VIC-MF6 coupling framework:
<img width="512" alt="Flowchart of the VIC-MF6 coupling framework" src="https://github.com/clawrim/VIC-MF6/blob/main/figures/vic-mf6-flowchart.png" />

This figure shows how to optimize the coupled distributed model:
<img alt="Distributed optimization of the VIC-MF6 coupling framework" src="https://github.com/clawrim/VIC-MF6/blob/main/figures/vic-mf6-optimization.gif" />

## Acknowledgments

This project is funded by the U.S. Geological Survey (USGS) Water Resources Research Act 104(b) grant [NM_2023_Cho](https://water.usgs.gov/wrri/grant-details.php?ProjectID=2023NM163B&Type=Annual) through the New Mexico Water Resources Research Institute (NM WRRI) under award GR0007017, as part of USGS Grant/Cooperative Agreement No. G21AP10635, along with an additional internal award from the NM WRRI.
