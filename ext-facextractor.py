import cv2
import os
import calendar
import time

# get running path
base_dir = os.path.dirname(__file__)

# Padding for recognized faces
SCALEY = 60
SCALEX = 60

# Status counters
total_files = 0
current_file = 0

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# Count total files for status
for filename in os.listdir('images'):
    total_files = total_files + 1

print('-----------------STARTED-----------------')

# Main Loop
for filename in os.listdir('images'):

    # Update status
    current_file = current_file + 1
    print('Image ' + str(current_file) + ' of ' + str(total_files))

    try:
        # read current image file
        img = cv2.imread(os.path.join(base_dir, 'images', filename))

        # pre processing
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.10, 8)

        # save detected faces
        for (x, y, w, h) in faces:
            crop_img = img[y-SCALEY:y+h+SCALEY, x-SCALEX:x+w+SCALEX]
            cv2.imwrite(os.path.join(base_dir, 'images_conv', str(calendar.timegm(time.gmtime())) + '.jpg'), crop_img)

            # wait one second to not overwrite the same file with timestamp
            time.sleep(1)
    except:
        print('Error: ' + filename)

print('-----------------FINISHED-----------------')