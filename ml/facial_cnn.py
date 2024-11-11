class FacialCnn:
        def __init__(self) -> None:
                

                pass                
        def train():        
# Deep Learning CNN model to recognize face
                '''This script uses a database of images and creates CNN model on top of it to test
                if the given image is recognized correctly or not'''

                '''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

                # Specifying the folder where images are present
                import time
                StartTime=time.time()

                TrainingImagePath='Final Training Images'

                from keras.preprocessing.image import ImageDataGenerator
                # Understand more about ImageDataGenerator at below link
                # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

                # Defining pre-processing transformations on raw images of training data
                # These hyper parameters helps to generate slightly twisted versions
                # of the original image, which leads to a better model, since it learns
                # on the good and bad mix of images
                train_datagen = ImageDataGenerator(
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True)

                # Defining pre-processing transformations on raw images of testing data
                # No transformations are done on the testing images
                test_datagen = ImageDataGenerator()

                # Generating the Training Data
                training_set = train_datagen.flow_from_directory(
                        TrainingImagePath,
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='categorical')


                # Generating the Testing Data
                test_set = test_datagen.flow_from_directory(
                        TrainingImagePath,
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='categorical')

                # Printing class labels for each face
                test_set.class_indices

                '''############ Creating lookup table for all faces ############'''
                # class_indices have the numeric tag for each face
                TrainClasses=training_set.class_indices

                # Storing the face and the numeric tag for future reference
                ResultMap={}
                for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
                        ResultMap[faceValue]=faceName

                # Saving the face map for future reference
                import pickle
                with open("ResultsMap.pkl", 'wb') as fileWriteStream:
                        pickle.dump(ResultMap, fileWriteStream)

                # The model will give answer as a numeric tag
                # This mapping will help to get the corresponding face name for it
                print("Mapping of Face and its ID",ResultMap)

                # The number of neurons for the output layer is equal to the number of faces
                OutputNeurons=len(ResultMap)
                print('\n The Number of output neurons: ', OutputNeurons)

                '''######################## Create CNN deep learning model ########################'''
                from keras.models import Sequential
                from keras.layers import Convolution2D
                from keras.layers import MaxPool2D
                from keras.layers import Flatten
                from keras.layers import Dense

                '''Initializing the neural network'''
                classifier= Sequential()

                classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))

                
                classifier.add(MaxPool2D(pool_size=(2,2)))

                
                classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

                classifier.add(MaxPool2D(pool_size=(2,2)))

                
                classifier.add(Flatten())

                
                classifier.add(Dense(64, activation='relu'))

                classifier.add(Dense(OutputNeurons, activation='softmax'))

                
                
                classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

                history = classifier.fit_generator(
                                training_set,
                                steps_per_epoch=8,
                                epochs=10,
                                validation_data=test_set,
                                validation_steps=10)
                
                classifier.save('model.h5')
                
                EndTime=time.time()
                last_accuracy = history.history['accuracy'][-1]

                time = "###### Total Time Taken: " + str(EndTime-StartTime) + ' seconds ######\n'
                accuracy = "###### Accuracy: " + str(last_accuracy)

                time = EndTime-StartTime
                accuracy = last_accuracy
                return {"time": time, "accuracy": accuracy}

        def predict(directory, image_file):

                '''########### Making single predictions ###########'''
                import numpy as np
                from keras.utils import load_img as image
                from keras.utils import img_to_array as img_to_array
                from keras.models import load_model
                import pickle
                import time as time

                StartTime=time.time()

                classifier = load_model('model.h5')
                ResultMap={}
                with open("ResultsMap.pkl", 'rb') as fileReadStream:
                        ResultMap = pickle.load(fileReadStream)

                ImagePath='Final Testing Images/'+directory+'/'+image_file
                test_image=image(ImagePath,target_size=(64, 64))
                test_image=img_to_array(test_image)

                test_image=np.expand_dims(test_image,axis=0)

                result=classifier.predict(test_image,verbose=0)
                
                predicted_class = np.argmax(result, axis=1)
                predicted_accuracy = result[0][predicted_class[0]]
                #print(training_set.class_indices)
                
                EndTime=time.time()

                return "Predicted face is: "+ResultMap[np.argmax(result)]+" with accuracy of " + str(predicted_accuracy) +" and time to predict of " + str(EndTime-StartTime)
                
