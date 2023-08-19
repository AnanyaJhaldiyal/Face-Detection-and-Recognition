#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import pickle
import os
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[2]:


import os
import imageio
import matplotlib.pyplot as plt
new_image_size = (150,150,3)
# set the directory containing the images
images_dir = './Headshots'
current_id = 0
label_ids = {}
images = []
labels = []
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith(('png','jpg','jpeg')):
            # path of the image
            path = os.path.join(root, file)
            # get the label name
            label = os.path.basename(root).replace(
                " ", ".").lower()

            # add the label (key) and its number (value)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # save the target value
            labels.append(current_id-1)
            # load the image, resize and flatten it
            image = imread(path)
            image = resize(image, new_image_size)
            images.append(image.flatten())

            # show the image
            plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
            plt.show()
print(label_ids)
# save the labels 
categories = list(label_ids.keys())
pickle.dump(categories, open("faces_labels.pk", "wb" ))
df = pd.DataFrame(np.array(images))
df['Target'] = np.array(labels)
df


# In[3]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.10,random_state=77, stratify = y)


# In[8]:


# trying out the various parameters
param_grid = {
    'C': [0.1, 1, 10, 100],'gamma' : [0.0001, 0.001, 0.1,1],'kernel' : ['rbf', 'poly']
}
svc = svm.SVC(probability=True)
print("Starting training, please wait ...")
# exhaustive search over specified hyper parameter
# values for an estimator
model = GridSearchCV(svc,param_grid)
model.fit(x_train, y_train)
# print the parameters for the best performing model
print(model.best_params_)


# In[9]:


y_pred = model.predict(x_test)


# In[10]:


pickle.dump(model, open('faces_model.pickle','wb'))


# In[23]:


import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
# loading the model and label
#pickle used to serialize data read binary
model = pickle.load(open('faces_model.pickle','rb'))
categories = pickle.load(open('faces_labels.pk', 'rb'))
# for detecting faces
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in range(1,40):
    test_image_filename = f'./facetest/president.jpg'
# load the image
imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
image_array = np.array(imgtest)
faces = facecascade.detectMultiScale(imgtest,scaleFactor = 1.1, minNeighbors = 5)
# if not exactly 1 face is detected,
# skip this photo
if len(faces)!= 1:
    print('---We need exactly 1 face; photo skipped---\n')
for (x_, y_, w, h) in faces:
    # draw the face detected
    face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h),(255, 0, 255), 2)
    plt.imshow(face_detect)
    plt.show()
    # resize and flatten the face data
    roi = image_array[y_: y_ + h, x_: x_ + w]
    img_resize = resize(roi, new_image_size)
    flattened_image = [img_resize.flatten()]
    # predict the probability
    probability =  model.predict_proba(flattened_image)
    for i, val in enumerate(categories):
        print(f'{val}={probability[0][i] * 100}%')
    print(f"{categories[model.predict(flattened_image)[0]]}")


# In[ ]:





# In[ ]:




