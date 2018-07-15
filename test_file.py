########################################
#### CODE BY SAI PRAJWAL KOTAMRAJU #####
########################################

from saliency import BMS_thresh
from findSalientRegions import find_sal_regions

import cv2

def drawRects(img, sal_regions):
    for reg in sal_regions:
        img = cv2.rectangle(img,(reg[0], reg[1]), (reg[2], reg[3]), (0,0,255), 2)
    cv2.imshow('Sal Regs', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    impath = 'images/men.jpg'
    img = cv2.imread(impath)
    sal_mask = BMS_thresh(impath,0.5) # Threshold for saliency map
    sal_regions = find_sal_regions(sal_mask,0.005) # Threshold for min area
    drawRects(img, sal_regions)                                           # of region
