import keras
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import rmsprop_v2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Create empty lists to store the training and test images
train_images = []
train_labels = []
test_images = []
test_labels = []

# Loop through the training folders
for folder in ['with_mask', 'without_mask']:
    # List all the file names in the current training folder
    files = os.listdir('Dataset/train/' + folder)

    # Loop through the file names
    for file in files:
        # Load the image
        img = cv2.imread('Dataset/train/' + folder + '/' + file)

        # Resize the image
        img = cv2.resize(img, (100, 100))

        # Convert the image to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # # Normalize the pixel values
        # mean, std = cv2.meanStdDev(img)
        # img = (img - mean) / std


        # Add the image to the list of training images
        train_images.append(img)

        # Add the corresponding label to the list of training labels
        if folder == 'with_mask':
            train_labels.append(1)
        else:
            train_labels.append(0)

# Repeat the same process for the test set
for folder in ['with_mask', 'without_mask']:
    files = os.listdir('Dataset/test/' + folder)
    for file in files:
        img = cv2.imread('Dataset/test/' + folder + '/' + file)
        img = cv2.resize(img, (100, 100))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mean, std = cv2.meanStdDev(img)
        # img = (img - mean) / std
        test_images.append(img)
        if folder == 'with_mask':
            test_labels.append(1)
        else:
            test_labels.append(0)



#print some images
# indices = [0,1,2,3,4]
#
# # Loop through the indices
# for i in indices:
#     # Get the image and its label
#     img = train_images[i]
#     label = train_labels[i]
#
#     # Print the label
#     print(label)
#
#     # Display the image
#     plt.imshow(img,cmap="gray")
#     plt.show()


# Convert the images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Reshape the images to have an additional channel dimension
# train_images = train_images.reshape((train_images.shape[0], 100, 100, 1))
# test_images = test_images.reshape((test_images.shape[0], 100, 100, 1))


# Normalize the pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0


# Load the base model
base_model = tf.keras.applications.VGG16(input_shape=(100, 100, 3), include_top=False, weights='imagenet')


# Add a few layers on top of the base model
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# freeze the layers of VGG
for layer in base_model.layers:
  layer.trainable = False

# Compile the model
#model.compile(optimizer=rmsprop_v2.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=5,
                    batch_size=16)


# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels)



# model=Sequential()
#
# model.add(Conv2D(100,(3,3),input_shape=train_images.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(100,(3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(1,activation='softmax'))




#
#
# # Define the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# model.add(tf.keras.layers.Dense(units=1, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# checkpoint = ModelCheckpoint(
#   'model-{epoch:03d}.model',
#   monitor='val_loss',
#   verbose=0,
#   save_best_only=True,
#   mode='auto')
#
# history=model.fit(
#   train_images,
#   train_labels,
#   epochs=20,
#   callbacks=[checkpoint],
#   validation_split=0.2)
#
# print(model.evaluate(test_images,test_labels))

def computeIntersectionOverUnion(bboxes, bboxes_GT):

    #This function computes the intersection over union score (iou)
    iou = 0

    #Compute the rectangle resulted by the intersection of the two bounding boxes
    # This should be specified in the following format [x1, y1, x2, y2]
    rectInters = [0, 0, 0, 0]


    for box in bboxes:
        for box_GT in bboxes_GT:


            assert box[0] < box[2]
            assert box[1] < box[3]
            assert int(box_GT[0]) < int(box_GT[2])
            assert int(box_GT[1]) < int(box_GT[3])

            # determine the coordinates of the intersection rectangle
            x_left = max(box[0], int(box_GT[0]))
            y_top = max(box[1], int(box_GT[1]))
            x_right = min(box[2], int(box_GT[2]))
            y_bottom = min(box[3], int(box_GT[3]))

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # Compute the area of rectInters (rectIntersArea)
            # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
            rectIntersArea = (x_right - x_left) * (y_top - y_bottom)


            # Compute the area of the box (boxArea)
            boxArea = (box[2] - box[0]) * (box[3] - box[1])

            # Compute the area of the box_GT (boxGTArea)
            boxGTArea = (int(box_GT[2]) - int(box_GT[0])) * (int(box_GT[3]) - int(box_GT[1]))

            # Compute the union area (unionArea) of the two boxes
            unionArea = boxArea + boxGTArea - rectIntersArea

            #Compute the intersection over union score (iou)
            iou = rectIntersArea / float(unionArea)

            print(iou)

            assert iou >= 0.0
            assert iou <= 1.0


    return iou
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def compareAgainstGT(imgf, bboxes):

    #This function compare the list of detected faces against the ground truth

    d_faces = 0         #the number of correctly detected faces
    md_faces = 0        #the number of missed detected faces
    fa = 0              #the number of false alarms

    bboxes_GT = []

    # Open the file with the ground truth for the associated image (imgf) and read its content

    files = glob.glob("./GT_FaceImages/*.txt")

    for f in files:
        c = open(f, "r", encoding="utf8")
        d = c.readlines()

        for elem in d:
            bboxes_GT.extend(elem.strip().split(';'))


    # Save the bounding boxes parsed from the GT file into the bboxes_GT list

    li = [s.split() for s in bboxes_GT]


    #Perform the validation of the bboxes (detected automatically)
    # against the bboxes_GT (annotated manually). In order to verify if two bounding boxes overlap it is necessary
    # to define another function denoted "computeIntersectionOverUnion(box, box_GT)"
    computeIntersectionOverUnion(bboxes, li)


    #Exercise 3 - Display the scores
    print("The scores for image {} are:".format(imgf))
    print("   - The number of correctly detected faces: {}".format(d_faces))
    print("   - The number of missed detected faces: {}".format(md_faces))
    print("   - The number of false alarms: {}".format(fa))


    return d_faces, md_faces, fa



def compareAgainstGT(imgf, bboxes):
    detection_threshold = 0.2

    #This function compare the list of detected faces against the ground truth

    d_faces = 0         #the number of correctly detected faces
    md_faces = 0        #the number of missed detected faces
    fa = 0              #the number of false alarms

    bboxes_GT = []

    # Open the file with the ground truth for the associated image (imgf) and read its content

    files = glob.glob("./GT_FaceImages/*.txt")

    for f in files:
        c = open(f, "r", encoding="utf8")
        d = c.readlines()

        for elem in d:
            bboxes_GT.extend(elem.strip().split(';'))


    #Save the bounding boxes parsed from the GT file into the bboxes_GT list

    li = [s.split() for s in bboxes_GT]


    #Perform the validation of the bboxes (detected automatically)
    # against the bboxes_GT (annotated manually). In order to verify if two bounding boxes overlap it is necessary
    # to define another function denoted "computeIntersectionOverUnion(box, box_GT)"
    iou = computeIntersectionOverUnion(bboxes, li)

    # Loop over the ground-truth bounding boxes
    for gt_bbox in li:
        # Check if the ground-truth box is detected
        is_detected = False
        for detected_bbox in li:
            iou = computeIntersectionOverUnion(gt_bbox, detected_bbox)
            if iou > detection_threshold:
                is_detected = True
                break
        if is_detected:
            d_faces += 1
        else:
            fa += 1
    # Loop over the detected bounding boxes
    for detected_bbox in li:
        # Check if the detected box is a false positive
        is_fp = True
        for gt_bbox in li:
            iou = computeIntersectionOverUnion(gt_bbox, detected_bbox)
            if iou > detection_threshold:
                is_fp = False
                break
        if is_fp:
            md_faces += 1


    # Display the scores
    print("The scores for image {} are:".format(imgf))
    print("   - The number of correctly detected faces: {}".format(d_faces))
    print("   - The number of missed detected faces: {}".format(md_faces))
    print("   - The number of false alarms: {}".format(fa))


    return d_faces, md_faces, fa


def extractCNNFeatures(croppedFaces):

    #Extract CNN features from cropped face images

    feats = []

    #Create a vggface model object
    # Load the VGG16 architecture pre-trained on the VGG Face dataset
    vgg_model = VGGFace(model='vgg16', input_shape=(224, 224, 3))

    last_layer = vgg_model.get_layer('pool5').output
    x = GlobalAveragePooling2D()(last_layer)

    model = Model(inputs=vgg_model.input, outputs=x)

    resizedFaces = []

    for crFace in croppedFaces:
        #Resize the image to (224, 224)
        resized_face = cv2.resize(crFace, (224, 224))
        resizedFaces.append(resized_face)


	#Convert resizedFaces to a float32 numpy array of size (n, 224, 224, 3)
        # Convert the list of resized faces to a numpy array
        face_array = np.array(resizedFaces)

        # Convert the data type of the array to float32
        face_array = face_array.astype('float32')

        # Normalize the pixel values to be between 0 and 1
        face_array /= 255.0

        # Add a batch dimension to the array
        face_array = np.expand_dims(face_array, axis=0)

    #Pre-process the face images to the standard format accepted by VGG16
    preprocessed_faces = preprocess_input(face_array)


	# Extract the low level features by forwarding the images through the CNN
    feats = model.predict(preprocessed_faces)



    return feats