# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:56:48 2020

@author: chang
"""
import numpy as np
import gdal
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Spectral_r):
    #plt.title(title)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #normalized confusion matrix
    #df_confusion = df_confusion / df_confusion.sum(axis=1)
    cax = ax.matshow(df_confusion, cmap=cmap, interpolation='none')
    fig.colorbar(cax)
    #show values, {:.2%}for percentage
    for (i, j), z in np.ndenumerate(df_confusion):
        ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    labels = ['Dry','Open water','Inundated vegetation']
    ax.set_xticklabels([''] + labels, rotation=45)
    ax.set_yticklabels([''] + labels)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel(df_confusion.columns.name)
    plt.ylabel(df_confusion.index.name)

    
def CalculateConfusionMatrix(Result1, Sample1, Result2, Sample2, Result3, Sample3):
    res_raster_1 = gdal.Open(Result1)
    res_arr_1 = np.array(res_raster_1.ReadAsArray())
    res_raster_2 = gdal.Open(Result2)
    res_arr_2 = np.array(res_raster_2.ReadAsArray())
    res_raster_3 = gdal.Open(Result3)
    res_arr_3 = np.array(res_raster_3.ReadAsArray())
    
    sam_raster_1 = gdal.Open(Sample1)
    sam_arr_1 = np.array(sam_raster_1.ReadAsArray())
    sam_raster_2 = gdal.Open(Sample2)
    sam_arr_2 = np.array(sam_raster_2.ReadAsArray())
    sam_raster_3 = gdal.Open(Sample3)
    sam_arr_3 = np.array(sam_raster_3.ReadAsArray())
    
    r_arr_1 = res_arr_1[np.where((sam_arr_1 == 0) | (sam_arr_1 == 1) | (sam_arr_1 == 2))]
    r_arr_2 = res_arr_2[np.where((sam_arr_2 == 0) | (sam_arr_2 == 1) | (sam_arr_2 == 2))]
    r_arr_3 = res_arr_3[np.where((sam_arr_3 == 0) | (sam_arr_3 == 1) | (sam_arr_3 == 2))]
    r_arr = np.concatenate([r_arr_1, r_arr_2,r_arr_3])
    print(r_arr.shape)
    
    s_arr_1 = sam_arr_1[np.where((sam_arr_1 == 0) | (sam_arr_1 == 1) | (sam_arr_1 == 2))]
    s_arr_2 = sam_arr_2[np.where((sam_arr_2 == 0) | (sam_arr_2 == 1) | (sam_arr_2 == 2))]
    s_arr_3 = sam_arr_3[np.where((sam_arr_3 == 0) | (sam_arr_3 == 1) | (sam_arr_3 == 2))]
    s_arr = np.concatenate([s_arr_1, s_arr_2,s_arr_3])
    print(s_arr.shape)
    
    y_actu = pd.Series(s_arr, name='Reference')
    y_pred = pd.Series(r_arr, name='Mapping results')
    df_confusion = pd.crosstab(y_actu, y_pred)
    #df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    plot_confusion_matrix(df_confusion)

if __name__ == '__main__':
    samples1 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\FieldSamples\FinalVersionProcessed\Yukon1-20170717-noBA-class3.tif'
    samples2 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\FieldSamples\FinalVersionProcessed\Yukon1-20170806-class3.tif'
    samples3 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\FieldSamples\FinalVersionProcessed\Yukon1-20170916-noBA-class3.tif'
    
    results1 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls\Inundationmap-2017-07-20.tif'
    results2 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls\Inundationmap-2017-08-01.tif'
    results3 = r'C:\Workspace\InundatedVegetationSatellite\S1Offline\Yukon1\GEEoutput-Desc\PostProcessing\Level2-InundationSeries-3cls\Inundationmap-2017-09-18.tif'
    
    CalculateConfusionMatrix(results1, samples1, results2, samples2, results3, samples3)
    
    print('done')