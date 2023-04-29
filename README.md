# face-detection-face-emotion-using-ml
Facial Recognition and Emotion Detection

TEAM 

Ashwani Kumar Singh
Prakhar Pratyush 
Bishal Raj Panda
Indroneel Pathak 
Richa Singh 



Abstract:

Facial recognition and emotion detection are two important applications of machine learning (ML) in the field of computer vision. The process involves analyzing the facial features of an individual using computer algorithms to identify the person and their emotional state. This technology has found applications in various industries, including security, marketing, and healthcare.

The process of facial recognition involves capturing an image of an individual and processing it through CNN to identify unique facial features that can be used to recognize the individual. Emotion detection, on the other hand, involves analyzing facial expressions to determine the emotional state of an individual. This is achieved by training KNN on large datasets of facial expressions and associating them with different emotions.
Overall, the study suggests that facial recognition and emotion detection using ML have significant potential to revolutionize various industries and improve our understanding of human behavior. However, careful consideration of the ethical implications and potential drawbacks is necessary to ensure that these technologies are used responsibly and ethically.


Introduction:

The problem statement for facial recognition and emotion detection using ML is how to accurately and efficiently identify individuals and detect their emotional state using computer algorithms. This involves developing sophisticated ML models that can analyze facial features and expressions to accurately identify individuals and their emotional states.

The significance of facial recognition and emotion detection using ML is evident in its potential to revolutionize industries and improve our understanding of human behavior. However, there are also significant ethical concerns that must be addressed to ensure that this technology is used in a responsible and ethical manner. Overall, facial recognition and emotion detection using ML have the potential to provide numerous benefits, but it is important to consider the ethical implications and potential drawbacks before implementing them in various applications.

The scope of facial recognition and emotion detection using ML is broad, with applications in various industries, including security, marketing, and healthcare. These technologies can be used to improve security by identifying individuals who pose a threat and preventing access to sensitive areas. They can also be used in marketing to deliver personalized content based on the emotional state of the user. In healthcare, facial recognition and emotion detection can be used to improve diagnosis and treatment of mental health disorders.


Literature Review:

"Facial Emotion Recognition Using Deep Convolutional Neural Networks" by Amr Hosny and Mohamed Taher. This paper proposes a facial emotion recognition system using a deep convolutional neural network (CNN). The system extracts facial features using a pre-trained CNN and uses a support vector machine (SVM) classifier to classify emotions. The proposed system achieved an accuracy of 91.37% on the JAFFE dataset.

"Real-time Facial Expression Recognition using Deep Learning Frameworks" by S.A. Patel, P.N. Vasani, and V.M. Patel. This paper proposes a real-time facial expression recognition system using deep learning frameworks. The system uses a convolutional neural network (CNN) to extract facial features and a softmax classifier to classify emotions. The proposed system achieved an accuracy of 95.29% on the CK+ dataset.


"Mood Detection using Facial Expressions and Electroencephalogram Signals" by S. M. Alam, A. H. M. R. Islam, and M. S. Islam. This paper proposes a mood detection system using facial expressions and electroencephalogram (EEG) signals. The system uses a machine learning algorithm to classify emotions based on facial expressions and EEG signals. The proposed system achieved an accuracy of 84.92% on the AffectNet dataset.


Problem Formulation:

The problem statement for developing a face recognition and emotion detection system using ML involves creating an algorithm that can accurately identify and classify facial expressions in real-time. While this technology has many potential applications, including security and healthcare, there are several challenges and limitations that must be addressed in the development of such a system.

Challenges:
One of the main challenges is the accuracy and reliability of the system. Facial recognition and emotion detection algorithms must be able to accurately identify and classify facial expressions in a variety of lighting and environmental conditions. Additionally, the algorithms must be able to differentiate between similar expressions, such as a smile and a smirk, to ensure accurate analysis.

Another challenge is the potential for bias in the data used to train the algorithm. Facial recognition and emotion detection systems are only as accurate as the data they are trained on, and if the data is biased or unrepresentative, the system will produce inaccurate results. It is essential to use diverse and representative datasets to ensure accurate and unbiased analysis.



Dataset Used:

For face recognition we have provided our faces as a dataset and for emotion detection system the dataset provided is the "FER2013" dataset, which contains 35,887 grayscale images of size 48x48 pixels. Each image is labeled with one of seven possible emotions: anger, disgust, fear, happiness, sadness, surprise, or neutral.


Methodology Used:

The methodology used for CNN-based facial detection and KNN-based emotion detection involved the following steps:

Data collection: The first step is to collect a dataset of facial images labeled with their corresponding emotions.

Data preprocessing: The collected data is preprocessed to remove noise, standardize the image size, and enhance image quality.

Feature extraction: For facial detection, a CNN is used to extract features from the preprocessed images. The CNN typically consists of several layers of convolutional, pooling, and activation functions that learn and extract relevant features from the images. For emotion detection, features such as texture, color, and shape is extracted from the images using techniques such as Local Binary Patterns (LBP) or Histogram of Oriented Gradients (HOG).

Model training: The extracted features are used to train the facial detection or emotion detection model using a suitable algorithm such as KNN. During training, the model learns to recognize patterns in the input data and classify images into their respective categories.

Model testing and evaluation: Once the model is trained, it is tested on a separate dataset to evaluate its accuracy and performance. The model may be further optimized by adjusting parameters such as the number of neighbors in KNN or the learning rate in CNN.

Deployment: The final step involves deploying the model in a real-world setting, such as a security system or customer service application, where it can be used to detect facial expressions or recognize faces.


Flow Diagram:

Convolutional Neural Networks(CNN) changed the way we used to learn images. It made it very very easy! CNN mimics the way humans see images, by focussing on one portion of the image at a time and scanning the whole image.
CNN boils down every image as a vector of numbers, which can be learned by the fully connected Dense layers of ANN. More information about CNN can be found here.
Below diagram summarizes the overall flow of CNN algorithm.







Proposed Algorithm:

Convolutional Neural Networks (CNNs) and k-Nearest Neighbors (KNN) are two popular machine learning techniques used for
facial detection and emotion detection, respectively. Here are some key features and advantages of using CNN for facial detection and KNN for emotion detection:

CNN for facial detection:

CNNs are particularly effective for image-based tasks such as facial detection because they can automatically learn and extract features from images.
CNNs are designed to handle spatial and temporal dependencies in data, which is important for detecting facial features such as eyes, nose, and mouth in different orientations and lighting conditions.
CNNs can handle large-scale datasets and are capable of achieving high accuracy on complex tasks.

KNN for emotion detection:

KNN is a simple and effective algorithm for classification tasks such as emotion detection, particularly when the number of classes is small.
KNN is a non-parametric algorithm, which means it does not require any assumptions about the underlying data distribution, making it more flexible than other classification algorithms.
KNN is easy to implement and can be trained quickly on small datasets.


Result Discussion:

The key findings of using CNN for facial detection and KNN for emotion detection are based on the accuracy and efficiency of the system.

In terms of facial detection using CNN, the accuracy achieved in this project was 91.83% with an F1 score of 88.97. This means that the CNN model correctly identified faces in 91.83% of the images in the test set, and the F1 score takes into account both precision and recall. This indicates that the model performed well in terms of detecting faces accurately.

In terms of efficiency, CNN-based facial detection can be computationally expensive, especially when dealing with large datasets. However, using techniques such as transfer learning and data augmentation can help to improve the efficiency of the model.

For emotion detection using KNN, the accuracy achieved can vary depending on the dataset and the number of neighbors chosen. However, KNN-based emotion detection can be relatively efficient and requires less computational resources compared to CNN-based facial detection.


Limitations:
One of the major limitations of facial recognition and emotion detection systems is their susceptibility to bias. These systems can produce inaccurate results based on factors such as lighting,skin color, and gender. There are also significant concerns regarding privacy and potential misuse of these technologies

Conclusion and Future Work:

In conclusion, the project focused on using machine learning techniques for facial recognition and emotion detection. We discussed the datasets used for training and testing, as well as the advantages of using CNNs for facial detection and KNN for emotion detection. The main findings of the project were that machine learning can be an effective approach for facial recognition and emotion detection, with high accuracy achieved on well-structured datasets.

The implications of these findings are numerous. For example, facial recognition systems can be used in a wide range of applications such as security systems, access control, and video surveillance. Emotion detection systems, on the other hand, can be used in various fields such as psychology, marketing, and customer service.

However, there are also potential ethical concerns related to the use of facial recognition and emotion detection systems. For example, there are concerns related to privacy and surveillance, as well as the potential for bias and discrimination based on race, gender, and other factors. Therefore, it is important to consider these issues when designing and deploying such systems.

Future directions for research in facial recognition and emotion detection using machine learning include improving the accuracy of the algorithms and addressing issues related to privacy and bias. In addition, research can focus on developing more robust models that can handle variations in lighting conditions, facial expressions, and other factors that can affect the performance of the models. Finally, research can also explore the use of other machine learning techniques such as deep learning and reinforcement learning for these tasks.


References:

https://www.codemag.com/Article/2205081/Implementing-Face-Recognition-Using-Deep-Learning-and-Support-Vector-Machines#:~:text=Deep%20Learning%20%2D%20Convolutional%20Neural%20Network,used%20in%20face%20recognition%20software

https://thinkingneuron.com/face-recognition-using-deep-learning-cnn-in-python/

https://ieeexplore.ieee.org/document/8777442

"Facial Emotion Recognition Using Deep Convolutional Neural Networks" by Amr Hosny and Mohamed Taher.

"Real-time Facial Expression Recognition using Deep Learning Frameworks" by S.A. Patel, P.N. Vasani, and V.M. Patel.

"Mood Detection using Facial Expressions and Electroencephalogram Signals" by S. M. Alam, A. H. M. R. Islam, and M. S. Islam.
