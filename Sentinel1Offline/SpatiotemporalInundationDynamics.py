'''
Created on Oct 17, 2020

@author: chuang71
'''

import os
import numpy as np
import gdal
import Utilities
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path, PurePath
from datetime import datetime
from dateutil.parser import parse

#get map list from Imagelist.txt
def GetMapList(MapList):
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
    return list_date
#get annual map list from Imagelist.txt
def GetAnnualMapList(MapList):
    list_annual_idx = {}
    f = open(MapList,'r')
    line = f.readline()
    while line:
        idx,bandinfo = line.split(':')
        idxx = int(idx)
        yyyy = bandinfo[7:11]
        list_annual_idx[yyyy].append(idxx)
    return list_annual_idx

def PlotSingleInundationSeries(InundationMapFolder):  
    datelist = []
    ow_sery =[]
    fm_sery = []
    fv_sery = []
    folder = Path(InundationMapFolder)
    dirs = folder.glob('Inundationmap-*.tif')
    tifffiles = [x for x in dirs if x.is_file()]
    for tif in tifffiles:
        inund = gdal.Open(str(tif))
        inund_arr = np.array(inund.ReadAsArray())
        
        ow_count = np.count_nonzero(inund_arr == 1)
        ow_area = ow_count * 100.0/1000000.0
        ow_sery.append(ow_area)
        fm_count = np.count_nonzero(inund_arr == 2)
        fm_area = fm_count * 100.0/1000000.0
        fm_sery.append(fm_area)
        fv_count = np.count_nonzero(inund_arr == 4)
        fv_area = fv_count * 100.0/1000000.0
        fv_sery.append(fv_area)
        
        mapname = os.path.basename(tif)
        mapdate = mapname[14:24]
        datelist.append(mapdate)
    
    count = len(datelist)
    ti = np.arange(0,count)

# stacked bar plot
    barWidth = 1
    bars = np.add(ow_sery, fm_sery).tolist()
    # Create ow bars
    p1 = plt.bar(ti, ow_sery, color='blue', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    p2 = plt.bar(ti, fm_sery, bottom=ow_sery, color='purple', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    p3 = plt.bar(ti, fv_sery, bottom=bars, color='green', edgecolor='white', width=barWidth)
     
    # Custom X axis
    xti = np.arange(0,count,2)
    d_list = [datelist[i] for i in xti]
    plt.xticks(xti, d_list, rotation=90)
    plt.ylabel('Area ($km^2$)')
    #plt.ylim(100,350) #for Yukon1
    plt.ylim(800,1300) #for PAD
    plt.legend((p1[0], p2[0], p3[0]), ('Open Water', 'Aquatic Vegetation', 'Flooded Vegetation'),loc='lower left')


# line plot    
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     ax1.plot(ti, ow_sery, color='blue', label = 'Open Water')
#     ax2.plot(ti, fm_sery, color='purple', label = 'Emergent Vegetation')
#     ax2.plot(ti, fv_sery, color='green', label = 'Flooded Vegetation')   
#     ax1.set_xticks(ti)
#     ax1.set_xticklabels(datelist, rotation=90)
#     ax1.set_ylabel('Open water area ($km^2$)')
#     ax2.set_ylabel('Wet vegetation area ($km^2$)')   
#     ax1.legend()
#     ax2.legend()
    
    
    plt.show()

def PlotSingleInundationSeriesRecls(InundationMapFolder):
    ow_sery =[]
    wet_sery = []
    ts = []
    folder = Path(InundationMapFolder)
    dirs = folder.glob('Inundationmap-*.tif')
    tifffiles = [x for x in dirs if x.is_file()]
    for tif in tifffiles:
        inund = gdal.Open(str(tif))
        inund_arr = np.array(inund.ReadAsArray())
        ow_count = np.count_nonzero(inund_arr == 1)
        ow_area = ow_count * 100.0/1000000.0
        ow_sery.append(ow_area)
        wet_count = np.count_nonzero(inund_arr == 2)
        wet_area = wet_count * 100.0/1000000.0
        wet_sery.append(wet_area)
        
        mapname = os.path.basename(tif)
        mapdate = mapname[14:24]
        ts.append(mapdate)
    
    count = len(ts)
    ti = np.arange(0,count)

# stacked bar plot
    barWidth = 1
    # Create ow bars
    p1 = plt.bar(ti, ow_sery, color='blue', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    p2 = plt.bar(ti, wet_sery, bottom=ow_sery, color='green', edgecolor='white', width=barWidth)
     
    # Custom X axis
    xti = np.arange(0,count,2)
    d_list = [ts[i] for i in xti]
    plt.xticks(xti, d_list, rotation=90)
    plt.ylabel('Area ($km^2$)')
    #plt.ylim(100,350) #for Yukon1
    plt.ylim(800,1300) #for PAD
    plt.legend((p1[0], p2[0]), ('Open Water', 'Inundated Vegetation'),loc='lower left')
    
# line plot    
#     fig, ax1 = plt.subplots()
# 
#     ax2 = ax1.twinx()
#     ax1.plot(ti, ow_sery, color='blue', label = 'Open Water')
#     ax2.plot(ti, wet_sery, color='purple', label = 'Wet Vegetation')
#     
#     ax1.set_xticks(ti)
#     ax1.set_xticklabels(ts, rotation=90)
#     ax1.set_ylabel('Open water area ($km^2$)')
#     ax2.set_ylabel('Wet vegetation area ($km^2$)')
#     ax1.legend()
#     ax2.legend()
    
    plt.show()        
  
def ReclassInundationMap(InitialFolder, ClassNum, OutFolder):   
    driver = gdal.GetDriverByName('GTiff')
    folder = Path(InitialFolder)
    dirs = folder.glob('Inundationmap-*.tif')
    tifffiles = [x for x in dirs if x.is_file()]
    print(tifffiles)
    for tif in tifffiles:
        print(tif)
        inund = gdal.Open(str(tif))
        inund_arr = np.array(inund.ReadAsArray())
        row, col = inund_arr.shape
        #print(inund_arr.shape)
        geotransform = inund.GetGeoTransform()
        geoprojection = inund.GetProjection()
    
        recls_arr = np.copy(inund_arr)
        #3cls: 0-dry, 1-ow, 2-wet veg; 4cls:0-dry, 1-ow, 2-emergent veg, 3-flooded veg
        if ClassNum == 3: 
            wet_idx = np.where((inund_arr == 2) | (inund_arr == 3) | (inund_arr == 4))
            recls_arr[wet_idx] = 2
        else:
            wet_idx = np.where((inund_arr == 2) | (inund_arr == 3))
            recls_arr[wet_idx] = 2
        outfn = os.path.basename(tif)
        print(outfn)
        ReclassedMap = os.path.join(OutFolder,outfn)
        Utilities.WriteToRasterFile(recls_arr, ReclassedMap, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)               

def MapInundationFrequency(InundationMapFolder, OutOWFrequency, OutFVFrequency, OutCombineFrequency):
    driver = gdal.GetDriverByName('GTiff')
    folder = Path(InundationMapFolder)
    dirs = folder.glob('Inundation*.tif')
    tifffiles = [x for x in dirs if x.is_file()]
    print(tifffiles)
    inundtemp = gdal.Open(str(tifffiles[0]))
    inundtemp_arr = np.array(inundtemp.ReadAsArray())
    row, col = inundtemp_arr.shape
    #print(inund_arr.shape)
    geotransform = inundtemp.GetGeoTransform()
    geoprojection = inundtemp.GetProjection()
    
    bands = len(tifffiles)
    ow_fre_arr = np.zeros((bands, row,col),np.byte)
    fv_fre_arr = np.zeros((bands, row,col),np.byte)
    cmb_fre_arr = np.zeros((bands, row,col),np.byte)
    b = 0
    for tif in tifffiles:
        inund = gdal.Open(str(tif))
        inund_arr = np.array(inund.ReadAsArray())

        single_arr_ow = np.zeros((row,col),np.byte)
        single_arr_fv = np.zeros((row,col),np.byte)
        single_arr_cmb = np.zeros((row,col),np.byte)
        ow_idx = np.where(inund_arr == 1)
        fv_idx = np.where(inund_arr == 2)
        cmb_idx = np.where(inund_arr != 0)
        single_arr_ow[ow_idx] = 1
        single_arr_fv[fv_idx] = 1
        single_arr_cmb[cmb_idx] = 1
        ow_fre_arr[b,:,:] = single_arr_ow
        fv_fre_arr[b,:,:] = single_arr_fv
        cmb_fre_arr[b,:,:] = single_arr_cmb
        b = b + 1
    
    out_ow_fre = np.nansum(ow_fre_arr, axis=0)
    out_fv_fre = np.nansum(fv_fre_arr, axis=0)
    out_cmb_fre = np.nansum(cmb_fre_arr, axis=0)
    
    Utilities.WriteToRasterFile(out_ow_fre, OutOWFrequency, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)     
    Utilities.WriteToRasterFile(out_fv_fre, OutFVFrequency, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)               
    Utilities.WriteToRasterFile(out_cmb_fre, OutCombineFrequency, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)                         

def GenerateMonthlyInundation(inFolder, outFolder):
    folder = Path(inFolder)
    dirs = folder.glob('Inundation*.tif')
    tifffiles = [x for x in dirs if x.is_file()]
    monthlist = {}
    for tif in tifffiles:
        tifname = PurePath(tif).name
        #print(tifname)
        yyyy = tifname[14:18]
        mm = tifname[19:21]
        monthlabel = yyyy + mm
        monthlist.setdefault(monthlabel,[]).append(tifname)
    
    print(monthlist)
    for key,value in monthlist.items():
        outfilename = 'Inundation-' + key + '.tif'
        outfile = os.path.join(outFolder, outfilename)
        filename0 = os.path.join(inFolder, value[0])
        inund0 = gdal.Open(filename0)
        inund_arr_0 = np.array(inund0.ReadAsArray())
        row, col = inund_arr_0.shape
        #print(inund_arr.shape)
        geotransform = inund0.GetGeoTransform()
        geoprojection = inund0.GetProjection()
        driver = gdal.GetDriverByName('GTiff')
        inund_arr_out = np.copy(inund_arr_0)
        if len(value) > 1:
            for tiffname in value[1:]:
                filename = os.path.join(inFolder, tiffname)
                inund = gdal.Open(filename)
                inund_arr = np.array(inund.ReadAsArray())
                inund_arr_out = np.maximum(inund_arr_out, inund_arr)
        
        Utilities.WriteToRasterFile(inund_arr_out, outfile, driver, col, row, 1, gdal.GDT_Byte, geotransform, geoprojection)     
        
        
        

if __name__ == '__main__':
    workfolder = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\PAD'  
    imglistfile = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\PAD\Imagelist.txt' #metadata of bands in L2file
    workspaceFolder = os.path.join(workfolder,'PostProcessing\Level2-InundationSeries-cleaned')
    Recls3Level2InundationFolder = os.path.join(workfolder, 'PostProcessing\Level2-InundationSeries-3cls')
    Recls4Level2InundationFolder = os.path.join(workfolder, 'PostProcessing\Level2-InundationSeries-4cls')
    if not os.path.exists(Recls3Level2InundationFolder):
        os.makedirs(Recls3Level2InundationFolder)
        os.makedirs(Recls4Level2InundationFolder)
        #reclass to 3 classes
        ReclassInundationMap(workspaceFolder, 3, Recls3Level2InundationFolder)
        ReclassInundationMap(workspaceFolder, 4, Recls4Level2InundationFolder)
    else:
        print('result folder exist, please check!')
    
    OutOWFrequencyMap = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls-owfrequency.tif'
    OutFVFrequencyMap = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls-wvfrequency.tif'
    OutCombineFrequencyMap = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls-inundfrequency.tif'
    # MapInundationFrequency(Recls3Level2InundationFolder, OutOWFrequencyMap, OutFVFrequencyMap, OutCombineFrequencyMap)
    
    PlotSingleInundationSeries(Recls4Level2InundationFolder)
    PlotSingleInundationSeriesRecls(Recls3Level2InundationFolder)
    
    MonthlyInundationFolder = os.path.join(workfolder, 'PostProcessing\MonthlyInundation-3cls')
    if not os.path.exists(MonthlyInundationFolder):
        os.makedirs(MonthlyInundationFolder)
        GenerateMonthlyInundation(Recls3Level2InundationFolder, MonthlyInundationFolder)
    else:
        print('monthly folder exist, please check!')
        
    outMonthOWFrequencyMap = os.path.join(MonthlyInundationFolder, 'owmonthfrequency.tif')
    outMonthFVFrequencyMap = os.path.join(MonthlyInundationFolder, 'fvmonthfrequency.tif')
    outMonthCOMFrequencyMap = os.path.join(MonthlyInundationFolder, 'monthfrequency.tif')
    # MapInundationFrequency(MonthlyInundationFolder, outMonthOWFrequencyMap, outMonthFVFrequencyMap, outMonthCOMFrequencyMap)
    
    print('done')