# WetlandInundationByS1
Tracking transient Arctic-Boreal wetland inundation from Sentinel-1 SAR

This repositry is the codes for mapping wetland inundation from Sentinel-1 C-band SAR imagery. This method was implemented mainly in Google Earth Engine (GEE), and requires post-processing offline using Python.

GEE code can be found in InundationMappingWithGEE.txt, which generates initial inundation maps based on time series of Sentinel-1 images on GEE.

Python code in /Sentinel1Offline/ provides post-processing functions to deal with the intial inundation maps dervied by GEE. Codes for mapping spatio-temporal inundation dynamics, and accuracy assessment are also included.

The corresponding paper has been published on GIScience & Remote Sensing. Please cite as below if you are using these codes.

Huang, C., L. C. Smith, E. D. Kyzivat, J. V. Fayne, Y. Ming, and C. Spence (2022), Tracking transient boreal wetland inundation with Sentinel-1 SAR: Peace-Athabasca Delta, Alberta and Yukon Flats, Alaska, GISci. Remote Sens., 59(1), 1767-1792, doi:10.1080/15481603.2022.2134620.
