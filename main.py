
# Importing necessary libraries
from keras.models import load_model
from keras.utils import img_to_array
import cv2
import numpy as np


#Importing the classifiers that we created and also a face_classifier to detect faces.
face_classifier = cv2.CascadeClassifier(r'C:\Users\Matil\ec_utbildning\ds23_deep_learning\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier_emotion =load_model(r'C:\Users\Matil\ec_utbildning\ds23_deep_learning\Emotion_Detection_CNN-main\history.h5')
classifier_age = load_model(r'C:\Users\Matil\ec_utbildning\ds23_deep_learning\age_gender.csv\age.h5')
classifier_ethnicity = load_model(r'C:\Users\Matil\ec_utbildning\ds23_deep_learning\age_gender.csv\ethnicity.h5')
classifier_gender = load_model(r'C:\Users\Matil\ec_utbildning\ds23_deep_learning\age_gender.csv\gender.h5')

# Setting the labels for emotion, ethnicity and gender classifiers so it will show in camera.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
ethnicity_labels = ['White', 'Black', 'Asian', 'Indian', 'Hispanic']
gender_labels = ['Male', 'Female']


# Initiating a camera module
cap = cv2.VideoCapture(0)

# Creating a while loop. If a face is detected the face is turned into grayscale. 
# We want the image to be gray because we trained our models on grayscale images.
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
 
# Creating a rectangle that appears around detected faces so we easily can recognise the face.
# We convert the image in the rectangle to 48 by 48 size, because this is the size of the pictures we trained models on.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

# We standardise the pixel values by dividing them with 255.
# We turn the values into an array because we need this input shape for the models to work.
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

# Now we apply our classifiers to the values in the array and thus get our predictions.
            prediction_emotion = classifier_emotion.predict(roi)[0]
            prediction_age = classifier_age.predict(roi)[0]
            prediction_ethnicity = classifier_ethnicity.predict(roi)[0]
            prediction_gender = classifier_gender.predict(roi)[0]

# Here we select the predicitons with the highest porbabilities.
            age = int(prediction_age[0])
            emotion = emotion_labels[prediction_emotion.argmax()]
            ethnicity = ethnicity_labels[prediction_ethnicity.argmax()]
            gender = gender_labels[prediction_gender.argmax()]

# We model the order of the output predicitons in a way that is suitable.
            label = f"{emotion}, {ethnicity}, {gender}, {age}"

# If we detect a face, the labels of predicitons will be printed out above the rectangle surrounding the face.
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# If we detect no faces, the text 'No faces' will be shown instead.       
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# We show our image.   
    cv2.imshow('Emotion Detector',frame)
    
# We add an exit command 'q' to be able to shut down the camera.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()