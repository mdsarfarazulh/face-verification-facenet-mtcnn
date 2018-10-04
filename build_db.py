'''
    @Author: Sarfarazul Haque.
    Here in this file a pickle databse is prepared using training dataset images.
    Each image associated with a label is feeded to the functions and an X array, having
    features of the face in the image, and a Y array, having all the labels associated with the 
    faces is stored to disk as feature_db.pickle.

    This is done to reduce the tesing time by having features of images calculated from before.
'''


import numpy as np 
import cv2
import os
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pickle


class build_DB(object):
    # This is a class having many functions to extract features and labels from the training images.

    def __init__(self):
        '''
            This function initializes the class parameter.
        '''

        # Creating a detector object of MTCNN class to detect faces.
	# For more details visit https://github.com/ipazc/mtcnn
        self.detector = MTCNN()

        # Setting shape of images.
        self.height = 160
        self.width = 160
        self.channels = 3

        # Loading Keras FaceNet model.
        self.model = load_model('models/facenet_keras.h5')

        # Setting training directory.
        self.train_dir = 'dev/train'

        # Setting pickle file path.
        self.p_file = 'data/feature_db.pickle'


    def extract_face(self, frame):
        '''
            This function extract faces from an image. It uses MTCNN pre-trained model. The extracted
            faces are stored in a list and returned to th calling function.

            @param1: frame: Frame from which the faces should be extracted.

            @return1: all_faces: A list of arrays of all faces detected on the current frame.
        '''

        # Calling detect_faces(frame) from MTCNN class through self.detector object.
        faces = self.detector.detect_faces(frame)
        all_faces = []

        # For each frame crop every face and append to all_faces list.
        for face in faces:
            # Getting the co-ordinates of the bounding box.
            x, y, w, h = face['box']
            # Resizing the Region Of Image(ROI).
            re_face = cv2.resize(frame[y:y+h, x:x+w], (self.height, self.width))
            # Appending extracted faces.
            all_faces.append(re_face)

        return all_faces


    def extract_features(self, frame):
        '''
            This function uses FaceNet model to extract features from each face present in the frame.

            @param1: frame: Frame currently under inspection.

            @return1: feature_map: Features extracted from each face.
        '''

        # Getting faces from the frame using self.extract_face(frame) helper function.
        # Converting dtype from np.uint8 to np.float16
        faces = np.array(self.extract_face(frame), dtype=np.float16)
        # Normalizing the data.
        faces /= 255.0
        # Extracting feature vector by applying FaceNet model
        feature_map = self.model.predict(faces)

        return feature_map


    def make_db(self):
        '''
            This is the driver function of this class.
            This function after extracting features from each training image, stores features 
            in a pickle file along with it's labels.

            Saves the pickle file in 'data' directory as 'feature_db.pickle'
        '''
        X = []
        Y = []

        file_names = os.listdir(self.train_dir)

        if len(file_names) == 0:
            print('The directory is empty!')
            return None

        for file in file_names:
            img_path = self.train_dir + '/' + file
            Y.append(file.split('.')[0])
            img = cv2.imread(img_path)
            X.append(self.extract_features(img))

        X = np.array(X)
        Y = np.array(Y)
        data = {'features' : X, 'labels': Y}

        with open(self.p_file, 'wb') as f:
            pickle.dump(data, f)
