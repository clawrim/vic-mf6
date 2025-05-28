#wd ~/hdd/nmhydro/coupling/VIC-MF6
#mkdir -p grassdata
#grass -c EPSG:26913 grassdata/vic_mf6_utm13n --text
#
## import coupling grid
#v.import input=spatial_data/coupling_grid.gpkg layer=vic_cells output=vic_uzf_cells
#v.import input=spatial_data/coupling_grid.gpkg layer=mf6_cells output=mf6_uzf_cells
#
#v.overlay ainput=mf6_uzf_cells binput=vic_uzf_cells output=mf6_vic_join oper=and --overwrite
#v.db.addcolumn mf6_vic_join columns="vic_id INTEGER, mf6_id VARCHAR(6), cpl_id VARCHAR(16), area_m2 DOUBLE PRECISION"
#v.to.db map=mf6_vic_join option=area column=area_m2 units=meters --overwrite
#db.execute sql="UPDATE mf6_vic_join SET vic_id = b_gridcel"
#db.execute sql="UPDATE mf6_vic_join SET mf6_id = substr('000' || CAST(a_row AS TEXT), -3, 3) || substr('000' || CAST(a_col AS TEXT), -3, 3)"
#db.execute sql="UPDATE mf6_vic_join SET cpl_id = mf6_id || '_' || vic_id"

#!/bin/sh
set -e

# minimal coupling workflow

echo "step 1: cd to project dir"
cd ~/hdd/nmhydro/coupling/VIC-MF6

echo "step 2: create grassdata"
mkdir -p grassdata

echo "step 3: create grass location"
grass -c EPSG:26913 grassdata/vic_mf6_utm13n --text

GISDB=grassdata
LOC=vic_mf6_utm13n
MAPSET=PERMANENT

echo "step 4: import vic cells"
grass $GISDB/$LOC/$MAPSET --exec \
  v.import input=spatial_data/coupling_grid.gpkg layer=vic_cells output=vic_uzf_cells --overwrite

echo "step 5: import mf6 cells"
grass $GISDB/$LOC/$MAPSET --exec \
  v.import input=spatial_data/coupling_grid.gpkg layer=mf6_cells output=mf6_uzf_cells --overwrite

echo "step 6: intersect layers"
grass $GISDB/$LOC/$MAPSET --exec \
  v.overlay ainput=mf6_uzf_cells binput=vic_uzf_cells output=mf6_vic_join oper=and --overwrite

echo "step 7: add id and area fields"
grass $GISDB/$LOC/$MAPSET --exec \
  v.db.addcolumn mf6_vic_join columns="vic_id integer,mf6_id varchar(6),cpl_id varchar(16),area_m2 double precision,area_ratio double precision" --overwrite

echo "step 8: compute intersection area"
grass $GISDB/$LOC/$MAPSET --exec \
  v.to.db map=mf6_vic_join option=area column=area_m2 units=meters --overwrite

echo "step 9: populate ids"
grass $GISDB/$LOC/$MAPSET --exec \
  db.execute sql="update mf6_vic_join set vic_id = b_gridcel"
grass $GISDB/$LOC/$MAPSET --exec \
  db.execute sql="update mf6_vic_join set mf6_id = substr('000' || cast(a_row as text),-3,3) || substr('000' || cast(a_col as text),-3,3)"
grass $GISDB/$LOC/$MAPSET --exec \
  db.execute sql="update mf6_vic_join set cpl_id = mf6_id || '_' || vic_id"

echo "step 10: compute mf6 cell area"
grass $GISDB/$LOC/$MAPSET --exec \
  v.to.db map=mf6_uzf_cells option=area column=mf6_area_m2 units=meters --overwrite

echo "step 11: join mf6 area"
grass $GISDB/$LOC/$MAPSET --exec \
  v.db.join map=mf6_vic_join column=a_cat other_table=mf6_uzf_cells other_column=cat subset_columns=mf6_area_m2 --overwrite

echo "step 12: compute ratio and export"
grass $GISDB/$LOC/$MAPSET --exec \
  db.execute sql="update mf6_vic_join set area_ratio = area_m2 / mf6_area_m2"
grass $GISDB/$LOC/$MAPSET --exec \
  v.out.ogr input=mf6_vic_join output=$(pwd)/mf6_vic_join.csv format=CSV --overwrite

echo "step 13: done, CSV written to $(pwd)/mf6_vic_join.csv"

echo "step 14; keep the last 6 (useful) columns"
awk -F, -vOFS=, '{for(i=NF-5;i<=NF;i++) printf "%s%s", $i, (i<NF?OFS:RS)}' input.csv > mf6_vic_join_last6.csv
