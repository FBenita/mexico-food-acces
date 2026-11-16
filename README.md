# Walking access to food in 92 Mexican metropolitan areas

Code for the paper "Walking to the supermarket or the street stall? The reversal of food access gradients in 92 Mexican cities" (Benita, F., submitted).

The pipeline measures walking distance over pedestrian street networks from every inhabited census block in Mexico's 92 metropolitan areas to the nearest food outlet under thirteen definitions. It then relates those distances to tract deprivation and vehicle availability, and writes every table and figure of the paper.

## Contents

| folder | content |
| --- | --- |
| `code/` | `mxo_utils.py`, the shared helpers every notebook imports: paths, crosswalks, key builders, census parsers, run logging |
| `notebooks/` | the pipeline, numbered in run order |
| `scripts/` | `clip_cem_to_metros.py`, which cuts the national elevation raster into one file per area, run once on a local machine |
| `paper_outputs/` | the LaTeX table fragments, figures and supplementary CSV files as written by notebook 12 for the submitted paper |

## Run order

| notebook | what it does | main outputs | runtime |
| --- | --- | --- | --- |
| 00 Pedestrian networks | builds the walking network of each area from the OpenStreetMap extract, keeping every connected component | `Networks/pedestrian_networks/*.gpkg` | about 2 h |
| 01 Network inventory | reads the 92 pedestrian networks and summarises length, intersections and connected components | `01_network_metrics.parquet` | 15 min |
| 02 Input inventory | records the structure of every input file | `02_input_inventory.parquet` | a few min |
| 03 Geographic frame | dissolves metropolitan boundaries and selects the census blocks inside them | `03_blocks.parquet`, `03_metro_boundaries.gpkg` | 1 h |
| 04 Census and deprivation | attaches population, dwellings, vehicles and both deprivation indices | `04_blocks.parquet`, `04_tracts.parquet` | 30 min |
| 05 Food outlets | classifies the establishment register into the destination sets | `05_outlets.parquet` | a few min |
| 06 Street commerce | reads the census street audit and flags blocks with stalls or vendors | `06_street_commerce.parquet` | a few min |
| 07 Block access | walking distance from every inhabited block to the nearest outlet under the thirteen definitions, plus the robustness sets | `07_block_access.parquet`, `07_block_access_robustness.parquet`, `07_network_attachment.parquet` | about 2 h |
| 08 Access and deprivation | tract and area measures and the estimation sample | `08_tract_access.parquet`, `08_area_access.parquet` | a few min |
| 10 Within and between | variance decomposition, rank agreement and within-area gradients | `10_*.parquet` | a few min |
| 11 Inference | the gradient models with clustered errors, contrasts and adjustments for multiple testing | `11_*.parquet` | 10 min |
| 13 Slope robustness | walking time over terrain with Tobler's hiking function, for the two anchor definitions | `13_block_slope.parquet`, `13_area_slope.parquet`, `13_slope_gradient.parquet` | 10 min |
| 12 Tables and figures | every table, figure and supplementary file of the paper | `Results/paper/` | 15 min |

Notebook 00 needs the osmium command-line tool, which it installs in Colab. Notebook 12 reads the output of notebook 13, so run 13 before 12. Each notebook caches its work by area, so an interrupted run resumes where it stopped. Each run writes a log, `Results/logs/run_log_<notebook>_<time>.md`, recording the environment, the input files, everything the notebook printed and the checks it passed.

## Inputs

None of the inputs are redistributed here. All are public, and the notebooks expect them under `Inputs/` in the project folder.

| folder | source | content |
| --- | --- | --- |
| `Inputs/Census/` | INEGI, Censo de Población y Vivienda 2020, Resultados por AGEB y manzana urbana | the 32 archives `ageb_mza_urbana_<entity>_cpv2020_csv.zip` |
| `Inputs/MGN/manzanas/` | INEGI, Marco Geoestadístico 2020 | `2020_1_00_M.zip` (blocks), `2020_1_00_A.zip` (urban tracts), `2020_1_00_E.zip` |
| `Inputs/CLEU2020/` | INEGI, Censo 2020, Características de las localidades y del entorno urbano | `Censo2020_CEU_eum_csv.zip` |
| `Inputs/DENUE/` | INEGI, Directorio Estadístico Nacional de Unidades Económicas | the six extracts for activity codes 461 to 469, 311 and 312, and 722 |
| `Inputs/Deprivation/` | CONAPO, Índice de marginación urbana 2020; CONEVAL, Grado de rezago social por AGEB urbana 2020 | `imu_2020.csv`, `GRS_AGEB_urbana_2020.xlsx` |
| `Inputs/metro_areas/` | SEDATU, CONAPO and INEGI, Metrópolis de México 2020 | one GeoPackage of municipalities per area, and the crosswalk `metros_92.csv` |
| `Inputs/DEM/` | INEGI, Continuo de Elevaciones Mexicano 4.0, 15 m | one raster per area, written by `scripts/clip_cem_to_metros.py` from the national raster |
| `Inputs/osm/` | OpenStreetMap contributors, Geofabrik extract of Mexico, snapshot of [date] | `mexico-latest.osm.pbf`; notebook 00 derives `mexico_walk.osm.pbf` from it |

OpenStreetMap changes every day, so networks rebuilt from a later extract will differ slightly from those used in the paper. The networks used are included in the data record.
