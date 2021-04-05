import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import correlate
from scipy.stats import spearmanr,pearsonr
from tensorflow.python.ops.gen_array_ops import rank_eager_fallback
from pipeline import process_img
from k_means_clustering import elbowM
from seg_img import find_nearest_col_rgb, find_ranges, sat_thresh_filter, rgb_hsv, segmentImg
from create_graph import find_nearest_col
from experiment_testing import create_filtered_dirs
from graph_classification import graph_classification

posRGB = {(255,0,0):'red', (0,128,0):'green', (0,0,255):'blue', (255,165,0):'orange', (255,0,255):'purple', (255,255,0):'yellow'}#, (128,128,128):'gray'}

def hsv_to_rgb(h,s,v):
    img = np.zeros((1,1,3), np.uint8)
    img[0,0]=(h,s,v)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img[0,0]

def parseAnnotations():
    with open('sample_train1/annotations.json') as f:
        annotations = json.load(f)

    myAnnotations = []
    line_indexes = []
    dot_indexes = []
    for annotation in annotations:
        remove_graph = False
        if ((annotation['type'] == 'line') or (annotation['type'] == 'dot_line')):# and (annotation['image_index'] == 442):
            myAnnotation = {}
            myAnnotation['index'] = annotation['image_index']
            myAnnotation['type'] = annotation['type']
            myAnnotation['x_axis'] = {}
            myAnnotation['x_axis']['label'] = annotation['general_figure_info']['x_axis']['label']['text']
            myAnnotation['x_axis']['bbox'] = annotation['general_figure_info']['x_axis']['label']['bbox']
            myAnnotation['y_axis'] = {}
            myAnnotation['y_axis']['label'] = annotation['general_figure_info']['y_axis']['label']['text']
            scale = annotation['general_figure_info']['y_axis']['major_ticks']['values']
            myAnnotation['y_axis']['bbox'] = annotation['general_figure_info']['y_axis']['label']['bbox']
            myAnnotation['title'] = {}
            myAnnotation['title']['text'] = annotation['general_figure_info']['title']['text']
            myAnnotation['title']['bbox'] = annotation['general_figure_info']['title']['bbox']
            myAnnotation['legend'] = {}
            myAnnotation['legend']['bbox'] = annotation['general_figure_info']['legend']['bbox']
            myAnnotation['legend']['items'] = {}
            allItems = annotation['general_figure_info']['legend']['items']
            if (len(allItems) < 4):# and (max(scale)<750 and min(scale)>-750): #possible remove the max scale stuff here
                myAnnotation['numSeries'] = len(allItems)
                for item in allItems:
                    legend_dict = {}
                    legend_dict['label'] = item['label']['text']
                    legend_dict['bbox'] = item['label']['bbox']
                    myAnnotation['legend']['items'][item['model']]=legend_dict
                if myAnnotation['type'] == 'line':
                    line_indexes.append(myAnnotation['index'])
                else:
                    dot_indexes.append(myAnnotation['index'])
                myAnnotation['series'] = {}
                rangelist = []
                for series in annotation['models']:
                    myAnnotation['series'][series['name']] = {}
                    #corr, _ = spearmanr(series['x'], series['y'])
                    corr = (series['y'][0]-series['y'][-1]) / (series['x'][0]-series['x'][-1])
                    print(corr)
                    myAnnotation['series'][series['name']]['numCorr'] = corr
                    if corr >= 0.10:
                        correlation = 'positive'
                    elif corr <= -0.10:
                        correlation = 'negative'
                    else:
                        correlation = 'neutral'
                    myAnnotation['series'][series['name']]['correlation'] = correlation
                    # convert hex to hsv and threshold
                    hex = series['color'].lstrip('#')
                    r,g,b = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
                    h,s,v = rgb_hsv(r,g,b)
                    bounds = find_ranges(h,s,v) # find and record the color ranges for each
                    for elem in bounds:
                        ((lh,ls,lv),(uh,us,uv)) = elem
                        rangelist.append(set(range(round(max(lh,0)),round(min(uh,180)))))
                    myAnnotation['series'][series['name']]['color'] = (h,s,v)
                    #imagePath = "sample_train1/" + "png/" + str(myAnnotation['index']) + ".png"
                    #segImg = segmentImg(imagePath)
                    #cols = [x[1] for x in segImg]
                    myAnnotation['series'][series['name']]['closest_col'] = find_nearest_col(hsv_to_rgb(h,255,255), posRGB)
                    if s<60:
                        remove_graph = True # look at this again might need else to reset to false
                if len(set.union(*rangelist)) != len([ele for sub in rangelist for ele in sub]): # check if there is overlap between the color ranges of the series in the graph
                    remove_graph = True
                    print('reason2')
                if not remove_graph:
                    myAnnotations.append(myAnnotation)
    #print(myAnnotations)
    indexes = []
    for annot in myAnnotations:
        indexes.append(annot['index'])
    print(len(myAnnotations))
    return myAnnotations

def selectImages(indexes):
    for index in indexes:
        path = "sample_train1" + "png/" + index + ".png"


def testK(annotations):
    ks = []
    ksReal = []
    indexes = []
    score = 0
    ctr = 0
    for graph in annotations:
        ctr += 1
        print(ctr)
        imagePath = "sample_train1/" + "png/" + str(graph['index']) + ".png"
        img = cv2.imread(imagePath)
        img = sat_thresh_filter(img,40)
        ks.append(elbowM(img)-1)
        ksReal.append(graph['numSeries'])
        indexes.append(graph['index'])
        # if ctr==20:
        #     break
    for i in range(len(ks)):
        if ks[i] == ksReal[i]:
            score +=1
    print(indexes)
    print(ks)
    print(ksReal)
    print(score/len(ks))
        
memoization = {}

def editfast(s,t):
    
    # change both to lower case to account for small case errors which are not important to distinguish between
    s = s.lower()
    t = t.lower()

    if (s,t) in memoization:
        return memoization[(s,t)]
    
    if s == "":
        return len(t)
    
    if t == "":
        return len(s)
    
    rtn = min([1 + editfast(s[1:], t), 1 + editfast(s, t[1:]), (s[ 0 ] != t[ 0 ]) + editfast(s[ 1 :], t[ 1 :])])
    
    memoization[(s,t)] = rtn
    
    return rtn

def testCorr(annotations):
    #annotations = annotations[:10]
    axis_hits = 0
    series_hits = 0
    hits = 0
    ctr = 0
    numGraphs = len(annotations)
    for graph in annotations:
        ctr += 1
        print(ctr)
        print("correlation pearson is: " + ','.join(str(graph['series'][x]['numCorr']) for x in graph['series']))
        imagePath = "sample_train1/png/" + str(graph['index']) + ".png"
        testAxis,testCorr = process_img(imagePath, use_text_not_color=False)
        display_string = "sample_train1/png/" + str(graph['index']) + ".png" + ", x axis: " + graph['x_axis']['label'] + ", y axis: " + graph['y_axis']['label'] + ", title: " + graph['title']['text']
        leg_set = set()
        label_to_corr_map = graph['series']
        print(label_to_corr_map)
        leg_display_str = ""
        for key in label_to_corr_map:
            leg_set.add(label_to_corr_map[key]['closest_col'] + ": " + label_to_corr_map[key]['correlation'])
            leg_display_str = leg_display_str + ", " + label_to_corr_map[key]['closest_col'] + ": " + label_to_corr_map[key]['correlation']
        print(display_string)
        print(leg_set)
        print(testAxis)
        print(testCorr)
        
        edds = editfast(display_string, testAxis)
        if edds < 6:
            axis_hits = axis_hits + 1
        else:
            print('axis fail --------------------------------')
        
        oldsh = series_hits
        for elem1 in leg_set:
            for elem2 in testCorr:
                memoization = {}
                edsm = editfast(elem1,elem2)
                if edsm < 3:
                    series_hits = series_hits + (1/len(leg_set))
                    break
        if oldsh == series_hits:
            print('series fail --------------------------------')
        if not bool(testCorr):
            print('empty set for legend')
        print("running series score tally: " + str(series_hits/ctr))
    hits = (axis_hits + series_hits)/2
    print("axis score: " + str(axis_hits/numGraphs))
    print("series score: " + str(series_hits/numGraphs))
    score = (hits/numGraphs)*100
    print("total score: " + str(score) + "%")


    return


def resave_QA_for_classification(annotations,train_val):
    for graph in annotations:
        imagePath = "sample_train1/png/" + str(graph['index']) + ".png"
        correlation = {}
        correlation = {graph['series'][x]['closest_col']: graph['series'][x]['correlation'] for x in graph['series']}
        segImg = segmentImg(imagePath)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
                fname = "QA_model" + "/" + train_val + "/" + corrstr + "/" + "seg_" + corrstr + "_" + closeCol + str(i) + "_graph" + str(graph['index']) + ".png" # changed to jpg
                plt.imsave(fname,img)
            else:
                print('closest color not found')
                print(col)
        plt.close('all')


def train_with_QA(annotations):
    cwd = os.getcwd()
    numGraphs = len(annotations)
    train_annot = annotations[:(4*(numGraphs//5))]
    val_annot = annotations[(4*(numGraphs//5)):]
    create_filtered_dirs("QA_model")
    resave_QA_for_classification(train_annot,"train")
    resave_QA_for_classification(val_annot,"validation")
    graph_classification(cwd,0,"wild_graphs_filtered")

#print(hsv_to_rgb(0,255,255))
annot = parseAnnotations()
#testK(annot)
testCorr(annot)
#train_with_QA(annot)