# import libraries
import matplotlib.pyplot as plt
import numpy as np
from random import choice as randomchoice
from numpy.random import randint
from numpy.random import random
from numpy.random import randn
from numpy.random import choice
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from seg_img import *
import math
import os
import csv
import math
import string

# create perturbed data
def genData(dataType):
    # determine variables
    if dataType == 'line':
        # slope and intercept
        sign = [-1,1]
        m = choice(sign) * random() # determine slope
        b = choice(sign) * randint(0,25) * random() # determine intercept
        delta = np.random.uniform(-50,50, size=(100,))
        X1 = np.arange(100)
        X2 = (m * X1) + b + delta
    elif dataType == 'bar':
        # slope and intercept
        sign = [-1,1]
        m = choice(sign) * random() # determine slope
        #b = choice(sign) * randint(0,50) * random() # determine intercept
        delta = np.random.uniform(-15,15, size=(50,))
        X1 = np.arange(50)
        X2 = (m * X1) + delta
        
        # # adjust intercept to make bar graph not cross y axis
        # minVal = min(X2)
        # #print(minVal)
        # b = 0 - minVal
        # X2 = (m * X1) + b + delta
    elif dataType == 'scatter':
        sign = [-1,1]
        correlation = choice(sign) * random()
        Y1 = randn(1000)
        Y2 = randn(1000)
        phi = (0.5) * math.asin(correlation)
        a = math.cos(phi)
        b = math.sin(phi)
        c = math.sin(phi)
        d = math.cos(phi)
        intercept = choice(sign) * randint(0,10) * random()
        X1 = (a * Y1) + (b * Y2)
        X2 = (c * Y1) + (d * Y2) + intercept
    else:
        print('error: gen_data -- no  graph type given')
        return

    # calculate correlation
    corr = corr, _ = spearmanr(X1, X2) # spearman correlation
    return (X1,X2,corr)

def col_dist(c1,c2):
    (r,g,b) = c1
    (R,G,B) = c2
    return math.sqrt((r-R)**2 + (g-G)**2 + (b-B)**2)

def find_nearest_col(color, dic):
    dist = list()
    (R,G,B) = color
    minD = 442 # max color distance
    minC = 'empty'
    for elem in dic:
        d = col_dist(elem,color) #math.sqrt((r-R)^2 + (g-G)^2 + (b-B)^2)
        if d < minD:
            minD = d
            minC = dic.get(elem)
    return minC
    
    
def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    res = ''.join(randomchoice(letters) for i in range(length))
    return res

# create multi data graph
def create_multiData(n, sN, train_val, seriesType, dcolor, dataStyle, model, pstyle='default', outputDir='graphs_filtered', outputDir2='images', noHard=False, verbose=0):
    print("noHard: ", noHard)
    #determine variables
    STcopy = seriesType
    possSeries = ['line', 'scatter', 'bar']
    varArr = np.empty (sN, tuple)
    for i in range (0,sN):
        if STcopy == 'random':
            STcopy = choice(possSeries)
        if STcopy == 'line':
            varArr[i] = (genData('line'),'line')
            # if noHard, ensure that all graphs are sufficently far away from the classification boundaries
            if noHard:
                while (abs(varArr[i][0][2]-0.4) < 0.15) or (abs(varArr[i][0][2]+0.4) < 0.15):
                    varArr[i] = (genData('line'),'line')
        elif STcopy == 'bar':
            varArr[i] = (genData('bar'),'bar')
            # if noHard, ensure that all graphs are sufficently far away from the classification boundaries
            if noHard:
                while (abs(varArr[i][0][2]-0.4) < 0.15) or (abs(varArr[i][0][2]+0.4) < 0.15):
                    varArr[i] = (genData('bar'),'bar')
        elif STcopy == 'scatter':
            varArr[i] = (genData('scatter'),'scatter')
            # if noHard, ensure that all graphs are sufficently far away from the classification boundaries
            if noHard:
                while (abs(varArr[i][0][2]-0.4) < 0.15) or (abs(varArr[i][0][2]+0.4) < 0.15):
                    varArr[i] = (genData('scatter'),'scatter')

        #print(varArr[i][0][2], "--------------------------------------------------")
        #print(abs(varArr[i][0][2]-0.4), abs(varArr[i][0][2]+0.4))
    # colors
    colors = ['r', 'g', 'b'] # add other colors to see if it affects learning
    posRGB = {(255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    if dcolor == 'multi2':
        colors = ['y', 'c', 'm']#colors = ['y', 'c', 'm', 'r', 'g', 'b']
        posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y'}#posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y', (255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}
    #colors = ['y', 'c', 'm', 'r', 'g', 'b']

    #posRGB = {(0,255,255):'c', (255,0,255):'m', (255,255,0):'y', (255,0,0):'r', (0,128,0):'g', (0,0,255):'b'}

    copyC = colors
    colArr = np.empty(sN, str)
    #colordermap = ['b','r','g'] 
    for i in range (0, sN):
        elem = choice(copyC)
        colArr[i] = elem
        #colArr[i] = colordermap[i] # THIS LINE IS ONLY FOR TRAINING MULTI PSTYLE STUFF REMOVE LATER
        copyC.remove(elem)


    # lineStyles
    lineStyles = ['solid', 'dotted', 'dashed', 'dashdot']
    LSarr = list()
    if dataStyle == 'multi':
        for i in range (0, sN):
            LSarr.append(choice(lineStyles))
    else:
        for i in range (0, sN):
            LSarr.append('solid')


    # plot
    label_to_corr_map = {}
    col_to_corr_map = {}
    correlation = {}
    fig, ax = plt.subplots()
    plot_options = plt.style.available
    if 'grayscale' in plot_options:
        plot_options.remove('grayscale')
    if 'dark_background' in plot_options:
        plot_options.remove('dark_background')
    plt.style.use('default')
    #pstyle = 'multi' # remove this later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if pstyle == 'multi':
        style = choice(plot_options)
        plt.style.use(style)
    longcol = {'r':'red', 'b':'blue', 'g':'green'}
    for i,var in enumerate(varArr):
        ((X1,X2,corrph),GT) = var
        #print(colArr[i], corrph)
        if corrph >= 0.4:
            correlation[colArr[i]] = 'positive'
        elif corrph <= -0.4:
            correlation[colArr[i]] = 'negative'
        else:
            correlation[colArr[i]] = 'neutral'
        lbl = get_random_string(randint(3,12))
        #print(correlation)
        label_to_corr_map[lbl] = correlation[colArr[i]]
        col_to_corr_map[longcol[colArr[i]]] = correlation[colArr[i]]
        #print(col_to_corr_map)
        if GT == 'line':
            if model == 'g':
                plt.plot(X1, X2, linestyle=LSarr[i], label=lbl, color=colArr[i])     
            else:
                plt.plot(X1, X2, linestyle=LSarr[i], label=lbl, color=colArr[i]) # color=colArr[i], 
        elif GT == 'scatter':
            if model == 'g':
                ax.scatter(X1, X2, label=lbl, color=colArr[i])     
            else:
                ax.scatter(X1, X2, label=lbl, color=colArr[i]) # color=colArr[i], 
        elif GT == 'bar':
            if model == 'g':
                w = 0.8
                ax.bar((len(varArr)*X1)+(w*i), X2, width=w, align='center', label=lbl, color=colArr[i])
            else:
                w = 0.8
                ax.bar((len(varArr)*X1)+(w*i), X2, width=w, align='center', label=lbl, color=colArr[i]) # color=colArr[i], 
        else:
            raise ValueError('graph type not recognized')
        
        #get color:
        #print(p[i].get_color())
        #col_to_corr_map[p[i].get_color()] = correlation[colArr[i]]

    # randomize label and title positions and  strings
    ylabelpos = ['left', 'right']
    xlabelpos = ['top', 'bottom']
    tpos = ylabelpos + ['center']
    X1s = get_random_string(randint(3,12))
    X2s = get_random_string(randint(3,12))
    titlestr = get_random_string(randint(3,12))
    plt.xlabel(X1s, labelpad=randint(2,10))
    plt.ylabel(X2s, labelpad=randint(2,10))
    ax.xaxis.set_label_position(choice(xlabelpos))
    ax.yaxis.set_label_position(choice(ylabelpos))
    leg = ax.legend()
    tobj = ax.set_title(titlestr,loc=choice(tpos))
    fig.canvas.draw()

    # name the given graph
    if model == 'g':
        fname = "placeholder.png"
        fig.savefig(fname)
        segImg = segmentImg(fname)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
                fname = outputDir + "/" + train_val + "/" + corrstr + "/" + "seg_" + corrstr + "_" + closeCol + str(i) + "_" + seriesType + "_graph" + str(n) + ".png" # changed to jpg
                plt.imsave(fname,img)
            else:
                print('closest color not found')
                print(col)
        plt.close('all')
    elif model == 's':
        fname = "series_filtered/" + train_val + "/" + str(sN) + "/" + str(sN) + "_graph" + str(n) + ".png" # changed to jpg
        fig.savefig(fname)
        img = cv2.imread(fname)
        img = sat_thresh_filter(img,30)
        im = Image.fromarray(img)
        im.save(fname)
        plt.close()
    elif model == 'exp10':
        fname = "placeholder.png"
        fig.savefig(fname)
        fname = outputDir2 + '/' + str(sN) + "_graph" + str(n) + ".png"
        segImg = segmentImg(fname)
        for i,(img,col) in enumerate(segImg):
            closeCol = find_nearest_col(col,posRGB)
            if closeCol in correlation:
                corrstr = correlation[closeCol]
        plt.close()

    else:
        # name the given graph
        fname = outputDir2 + "/" + "graph_" + str(n) + ".png"
        fig.savefig(fname)
        plt.close()
        if model == "pt":
            return ("graph_" + str(n), X1s, X2s, titlestr, label_to_corr_map, col_to_corr_map)
        else:
            return fname
    
    return ("line_graph" + str(n), float(corrph), "placeholder")


def create_filtered_dirs(dirname="graphs_filtered"):
    # create the appropriate training, validation, and correlation directories
        # ----------------------------------------

        # create graphs_filtered
        path = "./" + dirname #"./graphs_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        correlations = ["positive", "negative", "neutral"]
        for correlation in correlations:
            train_path = "./" + dirname + "/train/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./" + dirname + "/validation/" + correlation
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------

def create_training_data(size, graphType, color, dataStyle, outputDir="graphs_filtered", v=False):
    
    # create output directory
    create_filtered_dirs(outputDir)

    # match tuples
    (train_gt, val_gt) = graphType
    (train_col, val_col) = color
    (train_ds, val_ds) = dataStyle
    
    sNop = ["1", "2", "3"]
    for i in range (0,size):
        sNopC = int(choice(sNop))
        create_multiData(i+1, sNopC, "train", train_gt, train_col, train_ds, 'g', v)
    for i in range (0,size): # validate with less data
        sNopC = int(choice(sNop))
        create_multiData(i+1, sNopC, "validation", val_gt, val_col, val_ds, 'g', v)

    # print success to console
    print("create_training_data: complete")

    # return the current path (this will be used for the image classication program)
    return os.getcwd()




def train_series_class(size, dataStyle, directory):
    cwd=os.getcwd()
    if(cwd!=directory):
        print("error: create_data called from wrong directory")
    else:
        # ----------------------------------------
        path = "./series_filtered"
        try:
            os.mkdir(path)
        except OSError:
            print ("Warning: Creation of the directory %s failed, might already exist" % path)

        # create training and validation directories
        sNop = ["1", "2", "3"]
        for n in sNop:
            train_path = "./series_filtered/train/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)
            train_path = "./series_filtered/validation/" + n
            try:
                os.makedirs(train_path)
            except OSError:
                print ("Warning: Creation of the directory %s failed, might exist already" % train_path)

        # ----------------------------------------

        for i in range (0, size):
            sNopC = int(choice(sNop))
            create_multiData(i, sNopC, 'train', 'random', 'multi', dataStyle, 's')
            create_multiData(i, sNopC, 'validation', 'random', 'multi', dataStyle, 's')