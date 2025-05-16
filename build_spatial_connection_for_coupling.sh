wd ~/hdd/nmhydro/coupling/VIC-MF6
mkdir -p grassdata
grass -c EPSG:26913 grassdata/vic_mf6_utm13n --text

# import coupling grid
v.import input=spatial_data/coupling_grid.gpkg layer=vic_cells output=vic_uzf_cells
v.import input=spatial_data/coupling_grid.gpkg layer=mf6_cells output=mf6_uzf_cells

v.overlay ainput=mf6_uzf_cells binput=vic_uzf_cells output=mf6_vic_join oper=and --overwrite
v.db.addcolumn mf6_vic_join columns="vic_id INTEGER, mf6_id VARCHAR(6), cpl_id VARCHAR(16), area_m2 DOUBLE PRECISION"
v.to.db map=mf6_vic_join option=area column=area_m2 units=meters --overwrite
db.execute sql="UPDATE mf6_vic_join SET vic_id = b_gridcel"
db.execute sql="UPDATE mf6_vic_join SET mf6_id = substr('000' || CAST(a_row AS TEXT), -3, 3) || substr('000' || CAST(a_col AS TEXT), -3, 3)"
db.execute sql="UPDATE mf6_vic_join SET cpl_id = mf6_id || '_' || vic_id"
