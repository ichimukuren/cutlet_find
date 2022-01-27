import cv2
import numpy as np

def auto_canny(frame,sigma=0.33):
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(frame,lower,upper)

def find_circles(img,max_mean=0.3,circle_mean=0.05):
    """
        max_mean - Максимальное отличие радиусов. в процентах дефол 30%
        circle_mean - Максимальное отличие кол-ва вершин. строится градиент.
    """
    
    canny = auto_canny(img)
    canny2 = cv2.dilate(canny, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_radius = []
    for i,cnt in enumerate(contours):
        cnt = cv2.convexHull(cnt)
        radius = cv2.arcLength(cnt,True) / (2*np.pi)
        all_radius.append(radius)
    
    radius_average = np.average(all_radius,weights=all_radius)
    
    circles = []
    loss = lambda x: x if (x < 1) else x-1
    
    
    for i,cnt in enumerate(contours):
        cnt = cv2.convexHull(cnt)
        radius = cv2.arcLength(cnt,True) / (2*np.pi)
        if radius >= radius_average:
            ellipse = cv2.fitEllipse(cnt)
            if loss(np.average(ellipse[1],weights=ellipse[1])/np.mean(ellipse[1])) > circle_mean:
                continue
            
            (x,y),radius2 = cv2.minEnclosingCircle(cnt)
            circles.append(np.int0([x,y,np.mean([radius,radius2]),len(cnt)]))
    
    circles = np.array(circles)
    cntlen_average = np.average(circles[:,3],weights=circles[:,3])
    
    circles = circles[circles[:,3]/cntlen_average > 0.5]
    return circles[:,:3]