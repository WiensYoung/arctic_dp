# NSIDC-0116 Polar Pathfinder Sea Ice Motion Vectors v4

## Manual Download Instructions

This dataset is hosted at NSIDC DAAC and requires browser-based Earthdata OAuth2 login.
Automated download via earthaccess is not possible because the data is "on-prem" at NSIDC
(not in Earthdata Cloud).

### Steps:
1. Open: https://nsidc.org/data/nsidc-0116/versions/4
2. Click "Download Data" (top-right)
3. Sign in with Earthdata credentials: wiensyoung@gmail.com
4. Select date range: 2020-01-01 to 2020-12-31 (or narrower)
5. Select region: Arctic (Northern Hemisphere)
6. Download daily NetCDF files (.nc) to this directory

### Alternative (direct HTTPS with Earthdata token):
1. Get a bearer token from https://urs.earthdata.nasa.gov/
2. Use: curl -H "Authorization: Bearer <TOKEN>" \
     "https://daacdata.apps.nsidc.org/pub/DATASETS/NSIDC-0116.004/"

### Expected files (sample):
- 2020/icemotion_daily_nh_25km_20200101_v4.0.nc
- 2020/icemotion_daily_nh_25km_20200102_v4.0.nc
- ...

### Intended use:
Ice drift speed/direction priors for scenario calibration.
Core data for the ice force model (Lindqvist 1989).
