'''
    This project is based on two research papers.
    Citations{
        @article{
            1604.02878,
            Author = {Kaipeng Zhang and Zhanpeng Zhang and Zhifeng Li and Yu Qiao},
            Title = {Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks},
            Year = {2016},
            Eprint = {1604.02878},
            Eprinttype = {arXiv},
            Doi = {10.1109/LSP.2016.2603342},
        }

        @article{
            1503.03832,
            Author = {Florian Schroff and Dmitry Kalenichenko and James Philbin},
            Title = {FaceNet: A Unified Embedding for Face Recognition and Clustering},
            Year = {2015},
            Eprint = {1503.03832},
            Eprinttype = {arXiv},
            Doi = {10.1109/CVPR.2015.7298682},
        }
    }
    @Author: MD Sarfarazul Haque.

    Used two pre-trained model in this project, one is for face detection and
    another is for face verificaton.

    FaceNet and MTCNN are the two models.

    For more information about these two models give a shot to following git-repo:
        FaceNet: https://github.com/nyoki-mtl/keras-facenet.git
        MTCNN: https://github.com/ipazc/mtcnn.git

'''


import cv2
import numpy as np 
from build_db import build_DB
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import pickle
import os.path




class face_recog(object):
    # Main class of the project this class deals with the face recognition part of the project.

    def __init__(self):
        '''
            This function initializes all the class parameters.
        '''
        
        # Getting the object of build_DB class to deal with database.
        self.b_db = build_DB()
        # Loading the FaceNet model.
        self.model = load_model('models/facenet_keras.h5')
        # Setting the path of database file.
        self.p_file = 'data/feature_db.pickle'
        # Setting shape of each image.
        self.height = 160
        self.width = 160
        self.channels = 3

        # Getting the object of MTCNN() class to detect a face in an image.
        self.detector = MTCNN()

        # Setting the opencv parameters. 
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.color = (0, 255, 255)
        self.line = cv2.LINE_AA

        # Setting the threshold value.
        self.threshold = 15.0
        


    def get_euclidean(self, X, Y):
        '''
            This function calculates the euclidean distance or L2 distance between two images.
        '''
        return np.sqrt(np.sum(np.square(np.subtract(X, Y))))


    def load_data(self):
        '''
            This function loads data into X and Y array respectively
            X: This array contains features extracted from training samples
            Y: This array contains labels associated with those training samples
        '''

        # To check is there si a database of training sample present or not
        if not os.path.exists(self.p_file):
            # If database is not present then we have to create one
            db = build_DB()
            db.make_db()

        # Load tha database into our program.
        with open(self.p_file, 'rb') as f:
            data_dict = pickle.load(f)
        print('Loading data!')
        # Set the training data to self.X and self.Y
        self.X, self.Y = data_dict['features'], data_dict['labels']
        print('Data loaded:')
        


    def recog_each_face(self, face):
        '''
            This function recognize a single face, extracted from a single frame.

            @param1: face: The face extracted from the frame
                    it's shape is (self.height, self.width, self.channel)

            @return1: It returns the label predicted.
        '''

        # Reshaping the array into batch format so FaceNet model can work.
        face = face.astype(np.float16).reshape((1, self.height, self.width, self.channels))
        # Normalizing the data to reduce computing time and memory.
        face /= 255.0
        # Extracting feature vector.
        feature = self.model.predict(face)
        dist = []

        # Calculating euclidean distance.
        for s_x in self.X:
            dist.append(self.get_euclidean(s_x, feature))
        
        print(dist)
        dist = np.array(dist)
        # Getting the most similar face.
        indx = np.argmin(dist)

        if dist[indx] < self.threshold:
            return self.Y[indx]
        else:
            return "Opps!"


    def recog_each_frame(self, frame):
        '''
            This function deals with each frame and extract the faces from current frame and 
            sends those faces to self.recog_each_face(self, face) to generate label.

            @param1: frame: The frame currently being operated for face verification.

            @return1: It returns the frame with labeled associated with each face.
        '''

        # Uses MTCNN library to detect faces
        # For more information about this function below give a check to following git-repository
        # https://github.com/ipazc/mtcnn.git
        faces = self.detector.detect_faces(frame)

        # This for loop draws bounding box aroung the face with a name associated with the box.
        # If there is no face present in the frame then this function returns
        #  the original frame passed to it.
        for face in faces:
            # Getting the co-ordinates of the bounding box.
            x, y, w, h = face['box']
            # Getting Region Of Image(ROI)
            f_img = frame[y:y+h, x:x+w]
            # Resizing the face in the shape of (self.width, self.height)
            f_img = cv2.resize(f_img, (self.width, self.height))
            # Calling the helper function to get the label.
            label = self.recog_each_face(f_img)
            # Drawing rectangle and putting text on the bounding box of each fce
            cv2.rectangle(frame, (x,y), (x+w, y+h), self.color, 2, self.line)
            cv2.putText(frame, label, (x-3, y-3), self.font, 0.5, self.color, 1)

        return frame

    
    def test(self, mode, path, save_path=None):
        '''
            This is the driver function of this class.
            This function read the image or video content either from Webcam or File
            and sends each frame to self.recog_each_frame(self, frame) to get faces recognized
            and show each frame through cv2.imshow().

            @param1: mode: Whether the source is image or video
                    values are, `image`, `video`.

            @param2: path: Path to the file in case of video file or image mode or 
                            webcam number in case webcam mode. 

            @param3: save_path: If save_path != None and mode == `image` then save the output to
                            location specified by save_path.
        '''
    
        self.load_data()
        # If the reding mode selected is 'image'
        if mode == 'image':
            # Reading image file.
            img = cv2.imread(path)

            # To check if image is loaded or not.
            if len(img) == 0:
                print('No image found!')
                return
            
            frame = self.recog_each_frame(img)
            if save_path is not None:
                cv2.imwrite(save_path, frame)
            else:
                cv2.imshow('Frame', frame)
                cv2.waitKey(0)
            
        # If the reading mode selected is 'video'
        elif mode == 'video':
            # Opening a video source either file or webcam
            cap = cv2.VideoCapture(path)

            # Checking whether video source opened or not.
            if not cap.isOpened():
                print('Video not opened!')
                return

            # Operating until video source is present.
            while cap.isOpened():

                _, img = cap.read()
                frame = self.recog_each_frame(img)
                cv2.imshow('Frame', frame)

                k = cv2.waitKey(1) & 0xFF

                if k == 27 or k == ord('q'):
                    break
            
            # Releasing the video source.
            cap.release()
        # Destroying all the windows utilised by the app.
        cv2.destroyAllWindows()




face = face_recog()
face.test('image', 'dev/test/friends.jpg', 'output/friends.jpg')
