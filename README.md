# WetlandInundationByS1
Tracking transient Arctic-Boreal wetland inundation from Sentinel-1 SAR
This repositry is the codes for mapping wetland inundation from Sentinel-1 C-band SAR imagery. This method was implemented mainly in Google Earth Engine (GEE), and requires post-processing offline using Python.
GEE code can be found in InundationMappingWithGEE.txt, which generates initial inundation maps based on time series of Sentinel-1 images on GEE.
Python code in /Sentinel1Offline/ provides post-processing functions to deal with the intial inundation maps dervied by GEE. Codes for mapping spatio-temporal inundation dynamics, and accuracy assessment are also included.

The corresponding paper is currently under review by ISPRS Journal of photogrammetry and remote sensing. Please cite as below if you are using these codes.
Huang C, Smith L C, Kyzivat E D, et al. 2021. Tracking transient Arctic-Boreal wetland inundation from Sentinel-1 SAR. ISPRS Journal of photogrammetry and remote sensing, (Under review).
