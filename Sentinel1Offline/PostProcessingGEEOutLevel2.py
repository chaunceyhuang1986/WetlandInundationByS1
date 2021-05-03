'''
Created on Oct 17, 2020

@author: chuang71
'''
import os
import gdal
import numpy as np
from skimage import filters
from scipy import ndimage, spatial
import Utilities
from skimage.filters.rank import majority
from skimage.morphology import disk, ball

#generate cleanded inundation maps to separate tif files, as distance has been applied in level1 result, no distance restrictions will be applied here anymore
def CleanLevel2InundationMaps(InitialMap, MapList, CleanL1Map, PatchSizeLimit, CleanedMap):
    list_date = {}
    f = open(MapList,'r')
    line = f.readline()
    while line:
        idx,bandinfo = line.split(':')
        idxx = int(idx)
        yyyymmdd = bandinfo[7:17]
        list_date[idxx] = yyyymmdd
        line = f.readline()
    f.close()
    print(list_date)
    
    l1_inund = gdal.Open(CleanL1Map)
    l1_inund_arr = np.array(l1_inund.ReadAsArray())
    ini_dry_idx = np.where(l1_inund_arr == 0)
    
    inund = gdal.Open(InitialMap)
    inund_arr = np.array(inund.ReadAsArray())
    bands, row, col = inund_arr.shape
    print(inund_arr.shape)
    
    geotransform = l1_inund.GetGeoTransform()
    geoprojection = l1_inund.GetProjection()
    driver = gdal.GetDriverByName( 'GTiff' )
    
    #clean disconnected small patches
    # 3x3 structuring element with connectivity 2. a.k.a D-8
    struct = ndimage.generate_binary_structure(2, 2)
    
    for i in range(0,bands):
        out_arr_b = np.zeros_like(inund_arr[i,:,:])
        single_inund_arr = np.copy(inund_arr[i,:,:])
        single_inund_arr[ini_dry_idx] = 0 #set dry pixel in Level-1 map as dry
        ow_idx = np.where(single_inund_arr == 1)
        out_arr_b[ow_idx] = 1
        fm_idx = np.where(single_inund_arr == 2)
        out_arr_b[fm_idx] = 2
        av_idx = np.where(single_inund_arr == 3)
        out_arr_b[av_idx] = 3
        #clear small fv pixels
        tmp_arr_non_fv = np.zeros_like(single_inund_arr)
        fv_idx = np.where(single_inund_arr == 4)
        tmp_arr_non_fv[fv_idx] = 1 #make fv as 1, 0 as background 
        tmp_out_arr_fv = Utilities.filter_isolated_cells(tmp_arr_non_fv, struct, PatchSizeLimit)
        filtered_fv_idx = np.where(tmp_out_arr_fv == 1)
        out_arr_b[filtered_fv_idx] = 4
        
        out_arr = majority(out_arr_b, disk(3))
        
        mapname = os.path.join(CleanedMap, 'Inundationmap-' + list_date[i] + '.tif')
        print('mapping:' + list_date[i])
        
        Utilities.WriteToRasterFile(out_arr, mapname, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)
if __name__ == '__main__':
    workfolder = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1'  
    
    workspaceFolder = os.path.join(workfolder,'GEEoutput-Desc')
    
    L2filename = 'Level2-InundationSeries-initial.tif'
    L2file = os.path.join(workspaceFolder, L2filename)
    imglistfilename = 'Imagelist.txt'
    imglistfile = os.path.join(workspaceFolder, imglistfilename) #metadata of bands in L2file
    
    outWorkspace = os.path.join(workspaceFolder, 'PostProcessing')
    if not os.path.exists(outWorkspace):
        os.makedirs(outWorkspace)
        
    #level-2 mapping individual inundation map   
    CleanedInundMapName = 'Inundation-Level-1-clean.tif'
    CleanedInundMap = os.path.join(outWorkspace, CleanedInundMapName)         
    PatchSizeLimit = 9 
    CleanedLevel2InundationFolder = os.path.join(outWorkspace, 'Level2-InundationSeries-cleaned')
    if not os.path.exists(CleanedLevel2InundationFolder):
        os.makedirs(CleanedLevel2InundationFolder)
        CleanLevel2InundationMaps(L2file, imglistfile, CleanedInundMap, PatchSizeLimit, CleanedLevel2InundationFolder)
    else:
        print('result folder exist, please check!')
    print('done')