import urllib.request
import os
import re
import tarfile
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import cv2
def getUrlImgBoxes(mUrlImgBoxesDescription):
    html = str(urllib.request.urlopen(mUrlImgBoxesDescription).read())
    contents = re.findall(r'content=".*"',html)
    return "http://www.image-net.org"+contents[0][15:-1]

def downloadImages(mUrlImgPaths):
    html = str(urllib.request.urlopen(mUrlImgPaths).read())
    lines = html.split("\\r\\n")
    for line in lines:
        try:
            filename = line.split(" ")[0]
            url = line.split(" ")[1] 
            savePath = str(directory+"/images/"+filename+url[url.rfind('.'):len(url)])
            print(savePath)
            urllib.request.urlretrieve(url, savePath)
        except:
            print("image download error..")
    

wnid = "n06209940"
try:
    shutil.rmtree(wnid)
except:
    pass
mUrlImgPaths = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=" + wnid
mUrlImgBoxesDescription = "http://image-net.org/api/download/imagenet.bbox.synset?wnid=" + wnid
mUrlImgBoxes = ""
extractedFolder = "Annotation/"+wnid

directory = wnid
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory+"/images")
    os.makedirs(directory+"/Output")
# Get all xml bounded boxes file
mUrlImgBoxes = getUrlImgBoxes(mUrlImgBoxesDescription)
print(mUrlImgBoxes)
urllib.request.urlretrieve(mUrlImgBoxes, directory+"/data.tar.gz")
tar = tarfile.open(directory+"/data.tar.gz")
tar.extractall()
tar.close()
shutil.move(extractedFolder, directory)
os.rename(directory+"/"+directory,directory+"/boundedBoxesXml")
# Download image files..
print("Download images..")
downloadImages(mUrlImgPaths)
imgFiles = os.listdir(directory+"/images")

for imgFile in imgFiles:
    filename = imgFile.split(".")[0]
    print(imgFile)
    xmlBoundedBoxes = os.listdir(directory+"/boundedBoxesXml")
    for boxFile in xmlBoundedBoxes:
        #print(boxFile)
        tree = ET.parse(directory+"/boundedBoxesXml/"+boxFile)
        root = tree.getroot()
        currentXml = False
        width = -1
        height = -1
        xmin = -1
        ymin = -1
        xmax = -1
        ymax = -1
        xminList = []
        yminList = []
        xmaxList = []
        ymaxList = []
        for child in root:
            if(child.tag=="filename" and child.text==filename):
                print(child.text)
                currentXml = True
            if(child.tag=="size" and currentXml == True):
                for grandChild in child:
                    #print(grandChild.tag)
                    if(grandChild.tag=="width"):
                        width=grandChild.text
                        print(width)
                    if(grandChild.tag=="height"):
                        height=grandChild.text
                        print(height)
            if(child.tag=="object" and currentXml==True):
                for grandChild in child:
                    if(grandChild.tag=="bndbox"):
                        for grandGrandChild in grandChild:
                            if(grandGrandChild.tag=="xmin"):
                                xmin = int(grandGrandChild.text)
                                xminList.append(xmin)
                                print("xmin:"+str(xmin))
                            if(grandGrandChild.tag=="ymin"):
                                ymin = int(grandGrandChild.text)
                                yminList.append(ymin)
                                print("ymin:"+str(ymin))
                            if(grandGrandChild.tag=="xmax"):
                                xmax = int(grandGrandChild.text)
                                xmaxList.append(xmax)
                                print("xmax:"+str(xmax))
                            if(grandGrandChild.tag=="ymax"):
                                ymax = int(grandGrandChild.text)
                                ymaxList.append(ymax)
                                print("ymax:"+str(ymax))
        if(width!=-1):        
            try:         
                img = cv2.imread(directory+"/images/"+imgFile)  
                cv2.imwrite(directory+"/Output/clean_"+imgFile, img)
                height, width = img.shape[:2]
                blank_image1 = np.zeros((height,width,3), np.uint8)
                blank_image2 = np.zeros((height,width,3), np.uint8)
                blank_image3 = np.zeros((height,width,3), np.uint8)
                i = 0
                while i < len(xmaxList):   
                    xmin = xminList[i]
                    ymin = yminList[i]
                    xmax = xmaxList[i]
                    ymax = ymaxList[i]         
                    cv2.rectangle(img,(xmin,ymin),(xmin+(xmax-xmin),ymin+(ymax-ymin)),(0,255,0),2)
                    # Just 2 Points for rectangle  black white
                    cv2.rectangle(blank_image1,(xmin,ymin),(xmin,ymin),(255,255,255),2)
                    cv2.rectangle(blank_image1,(xmin+(xmax-xmin),ymin+(ymax-ymin)),(xmin+(xmax-xmin),ymin+(ymax-ymin)),(255,255,255),2)
                    # 4 Points for rectangle black white
                    cv2.rectangle(blank_image2,(xmin,ymin),(xmin,ymin),(255,255,255),2)
                    cv2.rectangle(blank_image2,(xmin+(xmax-xmin),ymin),(xmin+(xmax-xmin),ymin),(255,255,255),2)
                    cv2.rectangle(blank_image2,(xmin,ymin+(ymax-ymin)),(xmin,ymin+(ymax-ymin)),(255,255,255),2)
                    cv2.rectangle(blank_image2,(xmin+(xmax-xmin),ymin+(ymax-ymin)),(xmin+(xmax-xmin),ymin+(ymax-ymin)),(255,255,255),2)
                    # Full rectangle black white
                    cv2.rectangle(blank_image3,(xmin,ymin),(xmin+(xmax-xmin),ymin+(ymax-ymin)),(255,255,255),2)
                    i = i + 1
                cv2.imshow('image',img)
                #cv2.imshow("blank_image1",blank_image1)
                #cv2.imshow("blank_image2",blank_image2)
                #cv2.imshow("blank_image3",blank_image3)
                cv2.imwrite(directory+"/Output/boxed0_"+imgFile, img)
                cv2.imwrite(directory+"/Output/boxed1_"+imgFile, blank_image1)
                cv2.imwrite(directory+"/Output/boxed2_"+imgFile, blank_image2)
                cv2.imwrite(directory+"/Output/boxed3_"+imgFile, blank_image3)
                cv2.waitKey(1)
                print("==================")
            except:
                pass
                    



