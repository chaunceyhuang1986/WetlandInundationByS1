# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:40:17 2020

@author: chang
"""
import gdal
import numpy as np
import Utilities
import matplotlib.pyplot as plt 

def InundationFrequency(inFile, outFile):
    inund_raster = gdal.Open(inFile)
    inund_arr = np.array(inund_raster.ReadAsArray())
    geotransform = inund_raster.GetGeoTransform()
    geoprojection = inund_raster.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    band, row, col = inund_arr.shape
    new_arr = np.zeros_like(inund_arr)
    for b in range(0,band):
        arr = inund_arr[b,:,:]
        idx = np.where(arr == 2)
        n_arr = np.zeros_like(arr)
        n_arr[idx] = 1
        new_arr[b,:,:] = n_arr
    
    inund_fre = np.nansum(new_arr, axis=0)
    Utilities.WriteToRasterFile(inund_fre, outFile, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)  

def PlotComparisonWithJCR(DifferenceMap):
    raster_1 = gdal.Open(DifferenceMap)
    arr_1 = np.array(raster_1.ReadAsArray())
    row,col = arr_1.shape
    
    

if __name__ == '__main__':
    JCRfile = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\JCRWater\JCRMonthWaterSeries-2017-2019.tif'
    JCRFrequency = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\JCRWater\JCRMonthWaterFrequency.tif'
    InundationFrequency(JCRfile, JCRFrequency)
    
    JCRFrequency10m = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\PAD\JCRWater\JCRMonthWaterFrequency-10m.tif'
    OurFrequency = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\PAD\PostProcessing\MonthlyInundation-3cls\monthfrequency.tif'
    DifFrequencyMap = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\PAD\PostProcessing\MonthlyInundation-3cls\monthfrequency-jcr.tif'
    # PlotComparisonWithJCR(DifFrequencyMap)
    print('done')