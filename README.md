# face-verification-facenet-mtcnn

This is a project based on keras for face verification implementing One-Shot Learning using MTCNN for face detection on the image and FaceNet pre-trained model for calculating the eucledian embedding of an image.

The implementation of MTCNN is based on following paper:
<ul>
<li>
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li: “Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks”, 2016; <a href='http://arxiv.org/abs/1604.02878'>arXiv:1604.02878</a>. DOI: <a href='https://dx.doi.org/10.1109/LSP.2016.2603342'>10.1109/LSP.2016.2603342</a>.
</li>
  <li>
  Source for mtcnn can be downloaded from https://github.com/ipazc/mtcnn.
  </li>
</ul>

The implementation of FaceNet is based on following paper:
<ul>
<li>
Florian Schroff, Dmitry Kalenichenko: “FaceNet: A Unified Embedding for Face Recognition and Clustering”, 2015; <a href='http://arxiv.org/abs/1503.03832'>arXiv:1503.03832</a>. DOI: <a href='https://dx.doi.org/10.1109/CVPR.2015.7298682'>10.1109/CVPR.2015.7298682</a>.
</li>
<li>
The pre-trained model for keras can be downloaded from https://github.com/nyoki-mtl/keras-facenet
</li>
</ul>

During training period MTCNN detects faces from the images in dev/train and FaceNet model then find the euclidean embedding of these faces and these euclidean embeddings are then stored into a pickle file along with the names associated with the faces.

During testing period, either an image or a video source is provided to the algorithm and then MTCNN detects faces from these images or frames of the video source and then each face is embedded with some euclidean embedding using FaceNet model after that we calculate euclidean distances between each of the training embeddings and the embedding under test the respective distances are then thresholded and the least one among all the thresholded values is choosen as the corresponding person under inspection.
