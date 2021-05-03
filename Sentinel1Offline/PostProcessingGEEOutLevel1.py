'''
Created on Oct 16, 2020

@author: chuang71
'''
import os
import gdal
import numpy as np
from skimage import filters
from scipy import ndimage, spatial
import Utilities

#generate OW map from initial inundation map, as it has been cleaned use focal-median in GEE, we do not clean it here any more
def CleanOpenWaterMap(InitialLevel1InundationMap, CleanedOWMap):
    Inundmap = gdal.Open(InitialLevel1InundationMap)
    Inund_arr = np.array(Inundmap.ReadAsArray())
    geotransform = Inundmap.GetGeoTransform()
    geoprojection = Inundmap.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )
    print(Inund_arr.shape)
    row, col = Inund_arr.shape
    
    out_arr = np.zeros_like(Inund_arr)
    ow_idx = np.where((Inund_arr == 1) | (Inund_arr == 2))
    out_arr[ow_idx] = 1
    
    Utilities.WriteToRasterFile(out_arr, CleanedOWMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)

#calculate euclieand distance to open wate bodies
def CalculateMinEuclideanDistanceToIOW(InundationMap, EuclideanDistanceMap):        
    OWimg = gdal.Open(InundationMap)
    srcband = OWimg.GetRasterBand(1)

    drv = gdal.GetDriverByName('GTiff')    
    dst_ds = drv.Create(EuclideanDistanceMap, OWimg.RasterXSize, OWimg.RasterYSize, 1, gdal.GDT_Float32)
    
    dst_ds.SetGeoTransform(OWimg.GetGeoTransform())
    dst_ds.SetProjection(OWimg.GetProjectionRef())
    
    dstband = dst_ds.GetRasterBand(1)    
    #gdal.ComputeProximity(srcband, dstband, ["VALUES='1,2,3'"])
    gdal.ComputeProximity(srcband, dstband)

#generate IOW map from initial inundation map
def CleanIntermittentOpenWaterMap(InitialLevel1InundationMap, EuclideanDistanceMap, Distancelimit, CleanedIOWMap):
    Inundmap = gdal.Open(InitialLevel1InundationMap)
    Inund_arr = np.array(Inundmap.ReadAsArray())
    geotransform = Inundmap.GetGeoTransform()
    geoprojection = Inundmap.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )
    print(Inund_arr.shape)
    row, col = Inund_arr.shape
    
    ED = gdal.Open(EuclideanDistanceMap)
    ED_arr = np.array(ED.ReadAsArray())
    
    out_arr = np.zeros_like(Inund_arr)
    iow_idx = np.where(Inund_arr == 3)
    out_arr[iow_idx] = 1
    
    #eliminate fv patches far away from ow, still a little bit slow
    # 3x3 structuring element with connectivity 2. a.k.a D-8
    struct = ndimage.generate_binary_structure(2, 2)
    labeled_array_iow, num_features_iow = ndimage.label(out_arr, structure=struct)
    print(num_features_iow)
    minED_arr = np.array(ndimage.labeled_comprehension(ED_arr, labeled_array_iow, np.arange(1, num_features_iow+1), np.min, float, 0))
    print(minED_arr.shape)
    newArray = np.copy(labeled_array_iow)
    for i in np.arange(1, num_features_iow+1):
        newArray[labeled_array_iow == i] = minED_arr[i-1]
    area_mask = (newArray > Distancelimit)
    out_arr[area_mask] = 0
    
    Utilities.WriteToRasterFile(out_arr, CleanedIOWMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)    

#combine OW and IOW
def GenerateWaterMap(OWMap, IOWMap, AllOWMap):
    Inundmap = gdal.Open(OWMap)
    Inund_arr = np.array(Inundmap.ReadAsArray())
    geotransform = Inundmap.GetGeoTransform()
    geoprojection = Inundmap.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )
    #print(Inund_arr.shape)
    row, col = Inund_arr.shape
    
    IOW = gdal.Open(IOWMap)
    IOW_arr = np.array(IOW.ReadAsArray())
    
    out_arr = np.zeros_like(Inund_arr)
    iow_idx = np.where((Inund_arr == 1) | (IOW_arr == 1))
    out_arr[iow_idx] = 1
    Utilities.WriteToRasterFile(out_arr, AllOWMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)    

#clean TFV map from intial inundation map, and those disconnected to iow
def CleanTemperarilyFloodedVegetationMap(InitialLevel1InundationMap, EuclideanDistanceMap, Distancelimit, CleanedFVMap):
    Inundmap = gdal.Open(InitialLevel1InundationMap)
    Inund_arr = np.array(Inundmap.ReadAsArray())
    geotransform = Inundmap.GetGeoTransform()
    geoprojection = Inundmap.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )
    print(Inund_arr.shape)
    row, col = Inund_arr.shape
    
    ED = gdal.Open(EuclideanDistanceMap)
    ED_arr = np.array(ED.ReadAsArray())
    
    out_arr = np.zeros_like(Inund_arr)
    tfv_idx = np.where(Inund_arr == 4)
    out_arr[tfv_idx] = 1
   
    #eliminate fv patches far away from ow, still a little bit slow
    # 3x3 structuring element with connectivity 2. a.k.a D-8
    struct = ndimage.generate_binary_structure(2, 2)
    labeled_array_fv, num_features_fv = ndimage.label(out_arr, structure=struct)
    print(num_features_fv)
    minED_arr = np.array(ndimage.labeled_comprehension(ED_arr, labeled_array_fv, np.arange(1, num_features_fv+1), np.min, float, 0))
    print(minED_arr.shape)
    newArray = np.copy(labeled_array_fv)
    for i in np.arange(1, num_features_fv+1):
        newArray[labeled_array_fv == i] = minED_arr[i-1]
    area_mask = (newArray > Distancelimit)
    out_arr[area_mask] = 0
    
    Utilities.WriteToRasterFile(out_arr, CleanedFVMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)

#baed on cleaned components, generate clean level1 inundation map
def GenerateLevel1InundationMapClean(InitialLevel1InundationMap, CleanedOpenwaterMap, CleanedIntermittentOpenwaterMap, CleanedTemporarilyFloodedVegMap, CleanedInunadtionMap):
    inund = gdal.Open(InitialLevel1InundationMap)
    inund_arr = np.array(inund.ReadAsArray())
    row, col = inund_arr.shape
    print(inund_arr.shape)
    geotransform = inund.GetGeoTransform()
    geoprojection = inund.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )

    GOW = gdal.Open(CleanedOpenwaterMap)
    GOW_arr = np.array(GOW.ReadAsArray())
    IOW = gdal.Open(CleanedIntermittentOpenwaterMap)
    IOW_arr = np.array(IOW.ReadAsArray())
    TFV = gdal.Open(CleanedTemporarilyFloodedVegMap)
    TFV_arr = np.array(TFV.ReadAsArray())
    
    out_arr = np.copy(inund_arr)
    #for tfv pixels, if inund_arr !=4, make it =4
    ad_tfv_idx = np.where((TFV_arr == 1) & (inund_arr != 4))
    out_arr[ad_tfv_idx] = 4
    #for non-tfv pixels, if inund_arr ==4, make it=0 or 3
    ad_ntfv_idx_1 = np.where((TFV_arr == 0) & (inund_arr == 4) & (IOW_arr == 1))
    out_arr[ad_ntfv_idx_1] = 3
    ad_ntfv_idx_2 = np.where((TFV_arr == 0) & (inund_arr == 4) & (IOW_arr == 0))
    out_arr[ad_ntfv_idx_2] = 0
    #for iow pixels, if inund_arr != 1,2,3, make it =3
    ad_iow_idx = np.where((IOW_arr == 1) & ((inund_arr == 0) | (inund_arr == 4)))
    out_arr[ad_iow_idx] = 3
    #for non-iow pixels, if inund_arr == 1,2,3, make it =0 or 4
    ad_niow_idx_1 = np.where((IOW_arr == 0) & ((inund_arr != 0) & (inund_arr != 4)) & (TFV_arr == 0))
    out_arr[ad_niow_idx_1] = 0
    ad_niow_idx_2 = np.where((IOW_arr == 0) & ((inund_arr != 0) & (inund_arr != 4)) & (TFV_arr == 1))
    out_arr[ad_niow_idx_2] = 4
    #for gow pixels
    ad_gow_idx = np.where((GOW_arr == 1) & (inund_arr == 1))
    out_arr[ad_gow_idx] = 1
    ad_gow_idx_2 = np.where((GOW_arr == 1) & (inund_arr == 2))
    out_arr[ad_gow_idx_2] = 2 
    
    Utilities.WriteToRasterFile(out_arr, CleanedInunadtionMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)
if __name__ == '__main__':
    workfolder = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1'  
    
    workspaceFolder = os.path.join(workfolder,'GEEoutput-Desc')
    
    L1filename = 'Level1-InundationMap-Initial.tif'
    L1file = os.path.join(workspaceFolder, L1filename)

    outWorkspace = os.path.join(workspaceFolder, 'PostProcessing')
    if not os.path.exists(outWorkspace):
        os.makedirs(outWorkspace)
    
    #step-1: clean open water map    
    CleanedOWMapName = 'Inundation-Level-1-OW.tif'
    CleanedOWMap = os.path.join(outWorkspace, CleanedOWMapName)
    if not os.path.exists(CleanedOWMap):
        CleanOpenWaterMap(L1file, CleanedOWMap)
        
    #step-2: calculate euclidean distance from IOW pixels
    EDfilename = CleanedOWMapName[:-4] + '-ED.tif'
    EuclideanDistanceMap = os.path.join(outWorkspace, EDfilename)
    if not os.path.exists(EuclideanDistanceMap):
        CalculateMinEuclideanDistanceToIOW(CleanedOWMap, EuclideanDistanceMap)
    
    #step-3: generate cleaned IOW, and ensure IOW close to ow, and combine with ow    
    IntCleanedOWMapName = 'Inundation-Level-1-IOW.tif'
    IntCleanedOWMap = os.path.join(outWorkspace, IntCleanedOWMapName)
    AllWaterMap = os.path.join(outWorkspace, 'Inundation-Level-1-OW-All.tif')
    IOWDistanceLimit = 1 #in PIXEL
    if not os.path.exists(IntCleanedOWMap):
        CleanIntermittentOpenWaterMap(L1file, EuclideanDistanceMap, IOWDistanceLimit, IntCleanedOWMap)
        GenerateWaterMap(CleanedOWMap, IntCleanedOWMap, AllWaterMap)
    #step-3d: regenerate distance map fomr iow
    IOWEuclideanDistanceMap = os.path.join(outWorkspace, 'Inundation-Level-1-OW-All-ED.tif')
    if not os.path.exists(IOWEuclideanDistanceMap):
        CalculateMinEuclideanDistanceToIOW(AllWaterMap, IOWEuclideanDistanceMap)
        
    #step-4: clean temperarily flooded vegetation map
    CleanedTFVMapName = 'Inundation-Level-1-TFV.tif'
    CleanedTFVMap = os.path.join(outWorkspace, CleanedTFVMapName)
    DistanceLimit = 3 #in PIXEL
    if not os.path.exists(CleanedTFVMap):
        CleanTemperarilyFloodedVegetationMap(L1file, EuclideanDistanceMap, DistanceLimit, CleanedTFVMap)
    
    #step-5: generate cleaned inundationmap, may also delete some fv incorrectly
    CleanedInundMapName = 'Inundation-Level-1-clean.tif'
    CleanedInundMap = os.path.join(outWorkspace, CleanedInundMapName)
    if not os.path.exists(CleanedInundMap):
        GenerateLevel1InundationMapClean(L1file, CleanedOWMap, IntCleanedOWMap, CleanedTFVMap, CleanedInundMap)
    print('done!')