import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *  #import sort 
from datetime import datetime
import time

#Stream video from web cam
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

#time
today = datetime.now()

#Get video from file 
cap = cv2.VideoCapture("../media/test.mp4")

#Image mask
mask = cv2.imread('../media/mask.png')

save = cv2.VideoWriter('../output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30, (1280,720))
#Entry line 
line_x1_a, line_y1_a = 200, 0
line_x2_a, line_y2_a = 200, 720

#Exit line 
line_x1_b, line_y1_b = 1080, 0
line_x2_b, line_y2_b = 1080, 720

# cap.set(3,1280)
# cap.set(4,720)

car_positions = {}
current_car_speed = {}

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]




model = YOLO("Yolo-weights/yolov8n.pt")

limits = [400,297,673,297]
# Object Tracker 
tracker = Sort(max_age=30,min_hits=3, iou_threshold=0.3)
totalCars = []
avg_speed = []

carsFromLeft = []
carsFromRight = []
while True:
    current_date = today.strftime("%Y-%m-%d")
    current_time = today.strftime("%H:%M:%S")
    success, frame = cap.read()
#    maskedRegion = cv2.bitwise_and(frame,mask)
    frame_time = time.time()
    
    results = model(frame)

    #save.write(frame)
    detections = results[0].boxes.boxes

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1_t,y1_t,x2_t,y2_t = box.xyxy[0]
            x1,y1,x2,y2 = int(x1_t),int(y1_t),int(x2_t),int(y2_t)

            ## Get the image width, height. Using range between initial and final points
            w,h =  (x2-x1) , (y2-y1)

            ## Draw a cvzone rectangle around the image 
            #cvzone.cornerRect(img=frame,bbox=(x1,y1,w,h))
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), thickness=3)


            ## Get confidence level and show on image
            conf = math.ceil(box.conf[0] * 100) / 100
            
            #Get Object Name 
            classId = int(box.cls[0])
            className = coco_classes[classId]

            if conf > 0.5:
                ## Draw the label on the image
                label = f"{className}: {conf:.2f}"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1_label = max(y1, labelSize[1] + 10)
                cv2.rectangle(frame, (x1, y1_label - labelSize[1] - 10), (x1 + labelSize[0], y1_label + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, y1_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    #cv2.line(img=frame,pt1=(limits[0],limits[1]),pt2=(limits[2],limits[3]),color=(0,45,225),thickness=1)
    cv2.line(frame, (line_x1_a, line_y1_a), (line_x2_a, line_y2_a), (0, 255, 0), thickness=2)
    cv2.line(frame, (line_x1_b, line_y1_b), (line_x2_b, line_y2_b), (0, 0, 255), thickness=2)

    objectTrackerResult = tracker.update(detections[:,:5])
    for result in objectTrackerResult:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        w,h = x2 - x1, y2 - y1

        cx,cy = x1 + w // 2 , y1 + h // 2
        cv2.circle(frame,(cx,cy),1,(255,0,255),cv2.FILLED)

        if line_y1_b < cy < line_y2_b and line_x1_b - 15 < cx < line_x2_b + 15 :
            exit_time = frame_time
            if carsFromRight.count(id) == 0:
                carsFromRight.append(id)


        if  line_y1_a  < cy < line_y2_a  and line_x1_a - 15 < cx < line_x2_a + 15:
            entry_time = frame_time
            if carsFromLeft.count(id) == 0 :
                carsFromLeft.append(id)

        if(id in carsFromRight and id in carsFromLeft):
            totalCars = set(carsFromLeft + carsFromLeft)

        if id not in car_positions:
            car_positions[id] = {'position': (cx, cy), 'timestamp': frame_time, 'speed': 0}
            current_car_speed[id] = [0]
        else:
            # Calculate the distance and time difference
            prev_position = car_positions[id]['position']
            prev_timestamp = car_positions[id]['timestamp']
            distance = math.sqrt((prev_position[0] - cx) ** 2 + (prev_position[1] - cy) ** 2)
            time_diff = frame_time - prev_timestamp
            # Calculate the speed in pixels per second
            if distance > 5:

                speed = distance / time_diff
                avg_speed.append(speed)

                # Convert the speed from pixels per second to a more meaningful unit, e.g., km/h or mph
                # Based on the Assumption that the video represents an Urban District Street of Florida, USA
                # Where the average speed limit is 35 m/ph
                # Update the car position, timestamp, and speed
                conversion_factor =  35 / (sum(avg_speed)/len(avg_speed))
                car_speed_mgh = conversion_factor * speed
                current_car_speed[id].append(car_speed_mgh)
                car_speed = sum(current_car_speed[id])/len(current_car_speed[id])
                car_positions[id] = {'position': (cx, cy), 'timestamp': frame_time, 'speed': car_speed}
            else:
                car_positions[id] = {'position': (cx, cy), 'timestamp': frame_time, 'speed': 0}
            
        # Display the speed of the vehicle
        label = f'Speed : {car_positions[id]["speed"]:.2f}'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, labelSize[1] + 10)
        cv2.rectangle(frame, (x1, (y1_label-20) - labelSize[1] - 10), (x1 + labelSize[0], (y1_label-20) + baseLine - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (x1, (y1_label-20) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, f'Speed : {car_positions[id]["speed"]:.2f}', (50, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        
        #  ## Draw a cvzone rectangle around the image 
        #cvzone.cornerRect( frame, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,255))
        #cvzone.putTextRect(frame,text=f'{conf} - {coco_classes[classId]} id - {id}', pos=(max(0,x1),max(35,y1)))

    
    cv2.putText(frame,str(current_date),(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
    cv2.putText(frame,str(current_time),(50,80),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
    cv2.putText(frame,f'Vehicle Count : {len(totalCars)}',(50,110),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
    cv2.imshow("Image",frame)
    save.write(frame)
    cv2.waitKey(1)
   # cap.release()
   # save.release()

save.release()
