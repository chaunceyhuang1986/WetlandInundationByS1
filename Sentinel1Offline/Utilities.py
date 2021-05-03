'''
Created on Jan 13, 2020

@author: chuang71
'''
import sys
import csv
from datetime import date
import datetime
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
import math
from scipy import ndimage, spatial

#fill nodata value with its nearest neighbour 
def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """    
    if invalid is None: invalid = np.isnan(data)

    ind = ndimage.distance_transform_edt(invalid, 
                                    return_distances=False, 
                                    return_indices=True)
    return data[tuple(ind)] 

def filter_isolated_cells(array, struct, PatchSizeLimit):
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :param PatchSizeLimit: minimum patch size to be kept
    :return: Array with minimum region size > PatchSizeLimit
    """

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes < PatchSizeLimit)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array

def Kittler(im, out):
    #"""
    #The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    #Works on 8-bit images only
    #Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding

    h,g = np.histogram(im.ravel(),256,[0,256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p =  c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    out[:,:] = 0
    out[im >= t] = 255
 
# Hopkins statistics, evaluation before clustering
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def lee_filter(img, size):
    #Now mask all nans
    out = np.ma.fix_invalid(img, copy=False, fill_value=9999)
    out.set_fill_value(9999)
    
    img_mean = uniform_filter(out, (size, size))
    img_sqr_mean = uniform_filter(out**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(out)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (out - img_mean)
    return img_output
def WriteToRasterFile(out_array, out_name, Driver, Col, Row, Bandcount, DataType, Geotransform, Geoprojection):
    output_raster = Driver.Create(out_name, Col, Row, Bandcount, DataType) 
    if output_raster is None:
        print ("Could not create " + out_name)
        sys.exit(1)
        
    outBand = output_raster.GetRasterBand(1)
    # write the data
    outBand.WriteArray(out_array, 0, 0)
    
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-9999)

    output_raster.SetGeoTransform(Geotransform)
    output_raster.SetProjection(Geoprojection)
    return output_raster

#problem with projection
def WriteToRasterFile_multibands(out_array, out_name, Driver, Col, Row, Bandcount, DataType, Geotransform, Geoprojection):
    output_raster = Driver.Create(out_name, Col, Row, Bandcount, DataType) 
    if output_raster is None:
        print ("Could not create " + out_name)
        sys.exit(1)

    for i in range(Bandcount):    
        #print i
        outBand = output_raster.GetRasterBand(i + 1)
        # write the data
        outBand.WriteArray(out_array[i,:,:], 0, 0)
        
        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        outBand.SetNoDataValue(-9999)

    output_raster.SetGeoTransform(Geotransform)
    output_raster.SetProjection(Geoprojection)
    return output_raster
#get datelist of bands. id has been revised to be starting from 1.
def DateOfBand(list_txt):
    list_date = {}
    f = open(list_txt,'rb')
    line = f.readline()
    while line:
        idx,bandinfo = line.split(':')
        idxx = int(idx)
        yyyy = bandinfo[38:42]
        mm = bandinfo[43:45]
        dd = bandinfo[46:48]
        dayy = date(int(yyyy),int(mm),int(dd))
        list_date[dayy] = idxx
        line = f.readline()
    f.close()
    return list_date

def DateOfImage(imagelist_csv):
    list_date = {}
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            imgname,acdate = row[0], row[8]
            yyyy = acdate[0:4]
            mm = acdate[5:7]
            dd = acdate[8:10]
            list_date[imgname] = date(int(yyyy),int(mm),int(dd))
    return list_date  
def DateOfOrbit(imagelist_csv):
    list_date = {}
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            orbit,acdate = row[5], row[8]
            orb = orbit.zfill(5)
            yyyy = acdate[0:4]
            mm = acdate[5:7]
            dd = acdate[8:10]
            if orb not in list_date.keys():
                list_date[orb] = date(int(yyyy),int(mm),int(dd))
    return list_date  
def PathOfOrbit(imagelist_csv):
    list_path = {}
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            orbit,path = row[5], row[6]
            orb = orbit.zfill(5)
            pat = path.zfill(3)
            if orb not in list_path.keys():
                list_path[orb] = pat
    return list_path  
def OrbitsOfPath(imagelist_csv, path):
    output = set() #use set to ensure unique
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            orbit,p = row[5], row[6]
            orb = orbit.zfill(5)
            pat = p.zfill(3)
            if pat == path:
                output.add(orb)
    list_orbits = list(output)
    return list_orbits  
def FilenamesOfPath(imagelist_csv, path):
    list_filenames = []
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            granu,path = row[0], row[6]
            pat = path.zfill(3)
            if pat == path:
                fn = "AP_" + granu[6:11] + "_FBD_F" + granu[11:] + "_RT1"
                list_filenames.append(fn)
    return list_filenames   
def PathList(imagelist_csv):
    list_path = []
    with open(imagelist_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            path = row[6].zfill(3)
            if path not in list_path:
                list_path.append(path)
    return list_path 
def WaterlevelOfDate(waterlvel_csv):
    list_waterlevel = {}
    with open(waterlvel_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            acdate, wl = row[0], row[1]
            m,d,y=acdate.split("/")
            dt = date(int(y),int(m),int(d))
            list_waterlevel[dt] = float(wl)
            
    return list_waterlevel       
def SoilMoistureOfDate(soilmoisture_csv):
    list_soilmoisture = {}
    with open(soilmoisture_csv, 'r') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(imgreader)
        for row in imgreader:
            acdate, sm = row[0], row[1]
            m,d,y=acdate.split("/")
            dt = date(int(y),int(m),int(d))
            if sm != 'nan':
                list_soilmoisture[dt] = float(sm)
            
    return list_soilmoisture
#Fuzzy function for Z shape, using a single sample or an assembly (array) as input
def ZFuzzyFunction(Sample, MinMembership, MaxMembership, Z1, Z2):
    outMembership =  MaxMembership
    rang = MaxMembership - MinMembership
    if Sample <= Z1:
        outMembership =  MaxMembership
    elif Sample <= (Z1 + Z2)/2:
        outMembership =  MaxMembership - 2 * rang * ((Sample - Z1)/(Z2 - Z1)) * ((Sample - Z1)/(Z2 - Z1))
    elif Sample <= Z2:
        outMembership =  MinMembership + 2 * rang * ((Sample - Z2)/(Z2 - Z1)) * ((Sample - Z2)/(Z2 - Z1))
    else:
        outMembership = MinMembership
    return outMembership
#Fuzzy function for S shape, using a single sample or an assembly (array) as input
def SFuzzyFunction(Sample, MinMembership, MaxMembership, S1, S2):  
    outMembership =  MinMembership
    rang = MaxMembership - MinMembership
    if Sample <= S1:
        outMembership =  MinMembership
    elif Sample <= (S1 + S2)/2:
        outMembership =  MinMembership + 2 * rang * ((Sample - S1)/(S2 - S1)) * ((Sample - S1)/(S2 - S1))
    elif Sample <= S2:
        outMembership =  MaxMembership - 2 * rang * ((Sample - S2)/(S2 - S1)) * ((Sample - S2)/(S2 - S1))
    else:
        outMembership = MaxMembership
    return outMembership
def ZFuzzyFunctionArray(Sample_arr, MinMembership, MaxMembership, Z1, Z2):    
    outMembership_arr =  np.zeros_like(Sample_arr, np.float32)
    rang = MaxMembership - MinMembership
    idx_1 = np.where(Sample_arr <= Z1)
    outMembership_arr[idx_1] = MaxMembership
    idx_2 = np.where((Sample_arr > Z1) & (Sample_arr <= (Z1 + Z2)/2))
    outMembership_arr[idx_2] =  MaxMembership - 2 * rang * ((Sample_arr[idx_2] - Z1)/(Z2 - Z1)) * ((Sample_arr[idx_2] - Z1)/(Z2 - Z1))
    idx_3 = np.where((Sample_arr > (Z1 + Z2)/2) & (Sample_arr <= Z2))
    outMembership_arr[idx_3] =  MinMembership + 2 * rang * ((Sample_arr[idx_3] - Z2)/(Z2 - Z1)) * ((Sample_arr[idx_3] - Z2)/(Z2 - Z1))
    idx_4 = np.where(Sample_arr > Z2)
    outMembership_arr[idx_4] = MinMembership
    return outMembership_arr
def SFuzzyFunctionArray(Sample_arr, MinMembership, MaxMembership, S1, S2):      
    outMembership_arr =  np.zeros_like(Sample_arr, np.float32)
    rang = MaxMembership - MinMembership
    idx_1 = np.where(Sample_arr <= S1)
    outMembership_arr[idx_1] = MinMembership
    idx_2 = np.where((Sample_arr > S1) & (Sample_arr <= (S1 + S2)/2))
    outMembership_arr[idx_2] =  MinMembership + 2 * rang * ((Sample_arr[idx_2] - S1)/(S2 - S1)) * ((Sample_arr[idx_2] - S1)/(S2 - S1))
    idx_3 = np.where((Sample_arr > (S1 + S2)/2) & (Sample_arr <= S2))
    outMembership_arr[idx_3] =  MaxMembership - 2 * rang * ((Sample_arr[idx_3] - S2)/(S2 - S1)) * ((Sample_arr[idx_3] - S2)/(S2 - S1))
    idx_4 = np.where(Sample_arr > S2)
    outMembership_arr[idx_4] = MaxMembership
    return outMembership_arr 
def CalculateTxMatrix(TestSamples, ControlSample, ProbabilityBins): 
    BINCOUNT = np.size(ProbabilityBins) - 1
    Ec = np.size(ControlSample)
    Es = np.size(TestSamples)
    ci = math.ceil (Ec/BINCOUNT)
    clist = np.ones(BINCOUNT, np.float32) * ci
    clist[-1] = ci * BINCOUNT - Ec  
    slist = np.ones(BINCOUNT, np.float32)
    for i in range(0,BINCOUNT):
        lowlim = ProbabilityBins[i]
        uplim = ProbabilityBins[i+1]
        slist[i] = np.count_nonzero((TestSamples >= lowlim) & (TestSamples < uplim))
        
    c_list = clist/Ec
    s_list = slist/Es
    XX = (c_list - s_list) * (c_list - s_list)/(c_list + s_list)
    X2 = np.sum(XX)
    
    N = min(Ec, Es)
    Xpb = BINCOUNT/N
    sigmaXpb = math.sqrt(BINCOUNT)/N
    Tx = max(0, (X2 - Xpb)/sigmaXpb)
    
    c_list_2 = clist/(Ec * Ec)
    s_list_2 = slist/(Es * Es)
    fenmu = np.sqrt(c_list_2 + s_list_2)
    Zk = (c_list - s_list)/fenmu
 
    return Tx, Zk  
    
# generate probability bins with equal probability but different bin widths   
def ProbabilityBinning(Samples, BINCOUNT):
    Sam = np.sort(Samples) #sort from small to large numbers
    SamCount = math.ceil(np.size(Sam)/BINCOUNT)
    BINS = np.zeros(BINCOUNT+1)
    BINS[0] = Sam[0]
    BINS[-1] = Sam[-1]
    for i in range(1, BINCOUNT):
        BINS[i] = Sam[i*SamCount]
    return BINS                                                                                 