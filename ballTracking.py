import cv2
import sys
import pandas as pd
import math

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
## Read in detected bboxes

df_detections = pd.read_csv("bestBallProbabilities_resized.csv")
df_smallestDetections = pd.read_csv("smallestDetections_resized.csv")

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[-1]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    # elif tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT.create()

# Read video
#video = cv2.VideoCapture("C:/Users/andre/Documents/MLProjects/FootballAnalysis/Data/08fd33_4.mp4")
video = cv2.VideoCapture("C:/Users/andre/Documents/MLProjects/FootballAnalysis/resized_video.mp4")
#video = cv2.VideoCapture(0) # for using CAM
#video = cv2.resizeWindow(video, (960, 540)  )        # Resize image

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()

# Define an initial bounding box

bbox = (df_detections[df_detections["Frame"]=="frame0"]["bbox_xmin"].tolist()[0], df_detections[df_detections["Frame"]=="frame0"]["bbox_ymin"].tolist()[0], df_detections[df_detections["Frame"]=="frame0"]["bbox_width"].tolist()[0], df_detections[df_detections["Frame"]=="frame0"]["bbox_height"].tolist()[0])

# Uncomment the line below to select a different bounding box
#bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)
frameCounter = 0
distanceBetweenFrames = 0 
errorCounter = 0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        subsetDf= df_detections[df_detections["Frame"] == "frame"+str(frameCounter)]
        subsetDf_smallest= df_smallestDetections[df_smallestDetections["Frame"] == "frame"+str(frameCounter)]

        detectionProb = subsetDf["Probability"].tolist()[0] 
        frameCounter+=1
        detectedBBOX = [subsetDf["bbox_xmin"].tolist()[0],subsetDf["bbox_ymin"].tolist()[0],subsetDf["bbox_width"].tolist()[0],subsetDf["bbox_height"].tolist()[0]] 
        detectedBBOX_smallest = [subsetDf_smallest["bbox_xmin"].tolist()[0],subsetDf_smallest["bbox_ymin"].tolist()[0],subsetDf_smallest["bbox_width"].tolist()[0],subsetDf_smallest["bbox_height"].tolist()[0]] 

        # Tracking success
        if frameCounter >1:
            prevP1 = p1
            prevP2 = p2
            prevCenter = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
             
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        centerTracked = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
        centerDetected = [detectedBBOX[0]+0.5*detectedBBOX[2],detectedBBOX[1]+0.5*detectedBBOX[3]]
        
        if frameCounter >1:
            distanceBetweenFrames = math.dist(centerTracked,prevCenter)
        distanceBetween = math.dist(centerTracked,centerDetected)

        if  (distanceBetween < 20) and (distanceBetweenFrames <100 ) :

            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#ADD LOGIC TO TRY USING SMALLEST DETECTION BOX IF BIG ERROR WITH HIGHEST PROBABILITY
        elif (detectionProb > 0.55):
            tracker = cv2.legacy.TrackerCSRT.create()
            ret = tracker.init(frame,tuple(detectedBBOX))
            if not ret:
                print("ERROR UPDATING TRACKER")
            ok, bbox = tracker.update(frame)

            cv2.rectangle(frame, (detectedBBOX[0],detectedBBOX[1]),(detectedBBOX[0]+detectedBBOX[2],detectedBBOX[1]+detectedBBOX[3]),(0,255,0), 2, 1)
            cv2.rectangle(frame, (detectedBBOX_smallest[0],detectedBBOX_smallest[1]),(detectedBBOX_smallest[0]+detectedBBOX_smallest[2],detectedBBOX_smallest[1]+detectedBBOX_smallest[3]),(0,0,0), 2, 1)

        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            errorCounter +=1
            if errorCounter > 1000000:
                bbox = cv2.selectROI(frame, False)
                tracker = cv2.legacy.TrackerCSRT.create()
                ret = tracker.init(frame,detectedBBOX)
                if not ret:
                    print("ERROR UPDATING TRACKER")
                ok, bbox = tracker.update(frame)
                errorCounter =0

    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break
video.release()
cv2.destroyAllWindows()