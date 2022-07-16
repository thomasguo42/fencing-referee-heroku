
import cv2
import os

clip_path = "./data/"
image_path = "../data/fencing/images/"
def createImagesFromVideos(video_list):
    image_list = []
    for count, clip_name in enumerate(video_list):
        
        print(clip_name)
        # Playing video from file:
        cap = cv2.VideoCapture(clip_name)
        image_frame = 1
        currentFrame = 0
        clip_count = 0
        while(True):
            clip_count += 1
            # Capture frame-by-frame
            ret, frame = cap.read()
            #print("got frame: ", frame)
            # Saves image of the current frame in jpg file
            name = str(count) + "_" + str(clip_count) + '.jpg'
            print(image_path+name)
            if frame is None:
                break
            #cv2.imwrite(image_path+name, frame)
            image_list.append(frame)

            # To stop duplicate images
            currentFrame += 1

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    return image_list
