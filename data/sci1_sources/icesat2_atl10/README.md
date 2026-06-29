# ICESat-2 ATL10 Sea Ice Freeboard (V006)

## Manual Download Instructions

This dataset is hosted at NSIDC DAAC and requires browser-based Earthdata OAuth2 login.

### Steps:
1. Open: https://nsidc.org/data/atl10
2. Click "Download Data"
3. Sign in with Earthdata credentials: wiensyoung@gmail.com
4. Select version: V006 (latest)
5. Select date range: winter months only (Oct 2020 - Apr 2021)
   Note: ICESat-2 ATL10 is laser altimetry — only meaningful over ice-covered ocean
6. Select region: Arctic (latitude > 60N)
7. Download .h5 files to this directory

### Alternative (NSIDC API):
- API endpoint: https://n5eil02u.ecs.nsidc.org/egi/request
- Requires Earthdata bearer token in Authorization header
- Dataset: ATL10, version 006

### Important notes:
- ATL10 provides sea ice freeboard (not thickness)
- Freeboard-to-thickness conversion needed (multiply by ~7-10x based on hydrostatic equilibrium)
- Data is along-track (~17 m footprint), not gridded
- Only available 2018-present
- Each granule is ~2 GB; download only what you need

### Intended use:
High-resolution sea ice freeboard validation and freeboard-to-thickness calibration
against PIOMAS and CryoSat-2 thickness products.
