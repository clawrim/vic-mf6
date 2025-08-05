#!/bin/sh

set -e

#cd ~/hdd/nmhydro/coupling/VIC-MF6
## following two lines are required to run only once
## to create the workspace and import grids from both models.

#mkdir -p grassdata
#grass -c EPSG:26913 grassdata/vic_mf6_utm13n --text

#v.import input=spatial_data/coupling_grid.gpkg layer=vic_cells output=vic_uzf_cells --overwrite
#v.import input=spatial_data/coupling_grid.gpkg layer=mf6_cells output=mf6_uzf_cells --overwrite

# process starts here; useful to re run from here after changes
echo "step 1: intersect vic and mf6 cells"
v.overlay ainput=mf6_uzf_cells binput=vic_uzf_cells output=mf6_vic_join oper=and --overwrite

echo "step 2: add id and area fields"
v.db.addcolumn mf6_vic_join columns="vic_id integer, mf6_id varchar(6), cpl_id varchar(16), area_m2 double precision, mf6_area_ratio double precision" --overwrite

echo "step 3: compute intersection area"
v.to.db map=mf6_vic_join option=area column=area_m2 units=meters --overwrite

echo "step 4: populate IDs"
db.execute sql="update mf6_vic_join set vic_id = b_gridcel"
db.execute sql="update mf6_vic_join set mf6_id = substr('000' || cast(a_row as text), -3, 3) || substr('000' || cast(a_col as text), -3, 3)"
db.execute sql="update mf6_vic_join set cpl_id = mf6_id || '_' || vic_id"

echo "step 5: compute mf6 cell areas and join them"
v.to.db map=mf6_uzf_cells option=area column=mf6_area_m2 units=meters --overwrite
v.db.join map=mf6_vic_join column=a_cat other_table=mf6_uzf_cells other_column=cat subset_columns=mf6_area_m2 --overwrite

echo "step 6: compute mf6_area_ratio (portion of mf6 cell in vic)"
db.execute sql="update mf6_vic_join set mf6_area_ratio = area_m2 / mf6_area_m2"

echo "step 7: remove old vic_area_total if it exists"
db.execute sql="drop table if exists vic_area_total"

echo "step 8: compute total VIC area from all overlapping mf6"
db.execute sql="create table vic_area_total as select vic_id, sum(area_m2) as vic_area_m2 from mf6_vic_join group by vic_id"

echo "step 9: join VIC total area back to mf6_vic_join"
v.db.join map=mf6_vic_join column=vic_id other_table=vic_area_total other_column=vic_id subset_columns=vic_area_m2 --overwrite

echo "step 10: add vic_area_ratio column if needed"
v.db.addcolumn map=mf6_vic_join columns="vic_area_ratio double precision" --quiet

echo "step 11: compute vic_area_ratio (portion of vic cell in mf6)"
db.execute sql="update mf6_vic_join set vic_area_ratio = area_m2 / vic_area_m2"

echo "step 12: preview joined area ratios"
v.db.select mf6_vic_join columns=vic_id,mf6_id,area_m2,mf6_area_m2,vic_area_m2,mf6_area_ratio,vic_area_ratio separator=',' | head -n 10

echo "step 13: export to CSV"
v.out.ogr input=mf6_vic_join output=./mf6_vic_join_new.csv format=CSV --overwrite


## initial trial to establish the workflow. OBSOLETED

#mkdir -p grassdata
#grass -c EPSG:26913 grassdata/vic_mf6_utm13n --text
#v.import input=spatial_data/coupling_grid.gpkg layer=vic_cells output=vic_uzf_cells
#v.import input=spatial_data/coupling_grid.gpkg layer=mf6_cells output=mf6_uzf_cells
#v.overlay ainput=mf6_uzf_cells binput=vic_uzf_cells output=mf6_vic_join oper=and --overwrite
#v.db.addcolumn mf6_vic_join columns="vic_id INTEGER, mf6_id VARCHAR(6), cpl_id VARCHAR(16), area_m2 DOUBLE PRECISION"
#v.to.db map=mf6_vic_join option=area column=area_m2 units=meters --overwrite
#db.execute sql="UPDATE mf6_vic_join SET vic_id = b_gridcel"
#db.execute sql="UPDATE mf6_vic_join SET mf6_id = substr('000' || CAST(a_row AS TEXT), -3, 3) || substr('000' || CAST(a_col AS TEXT), -3, 3)"
#db.execute sql="UPDATE mf6_vic_join SET cpl_id = mf6_id || '_' || vic_id"

