#!/usr/bin/env python
import pylab
import numpy as np
import os
colorCounter = 11
color_list = pylab.cm.Set1(np.linspace(0, 1, 26))


def getColorLabel(name):
    print (name)
    global colorCounter, color_list
    if name.find("SquaresChnFtrs") != -1 or name.find("Baseline") != -1:
        color = 'black'
        label = "SquaresChnFtrs-5"

    elif name.find("Ours_Headhunter") != -1 or name.find("Ours HeadHunter") != -1:
        color = 'r'
        label = "Ours HeadHunter"
    elif name.find("Face++") != -1:
        color = 'b'
        label = "Face++"
    elif name.find("Picasa") != -1:
        color = 'r'
        label = "Picasa"
    elif name.find("Structured") != -1:
        color = color_list[10]
        label = "Structured Models"
    elif name.find("WS_Boosting") != -1:
        color = color_list[1]
        label = "W.S. Boosting [14]"
    elif name.find("Sky") != -1:
        color = color_list[2]
        label = "Sky Biometry [28]"
    elif name.find("OpenCV") != -1:
        color = color_list[3]
        label = "OpenCV"
    elif name.find("TSM") != -1:
        color = color_list[4]
        label = "TSM"
    elif name.find("DPM") != -1 or name.find("<0.3") != -1:
        color = color_list[8]
        #color = 'b'
        label = "DPM"
    elif name.find("Shen") != -1:
        color = color_list[7]
        #color = 'b'
        label = "Shen et al."
    elif name.find("Viola") != -1:
        color = color_list[9]
        #color = 'b'
        label = "Viola Jones"
    else:
        color = color_list[colorCounter]
        colorCounter = colorCounter + 1
        label = os.path.splitext(os.path.basename(name))[0]
        label = label.replace("_", " ")
    return [color, label]
