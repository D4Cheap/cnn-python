class FacialCnn:
        def __init__(self) -> None:
                

                pass                
        def train():        
# Deep Learning CNN model to recognize face
               

                # Specifying the folder where images are present
                import time
                StartTime=time.time()

                TrainingImagePath='Final Training Images'

                from keras.preprocessing.image import ImageDataGenerator
                
                train_datagen = ImageDataGenerator(
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True)

                
                test_datagen = ImageDataGenerator()

               
                training_set = train_datagen.flow_from_directory(
                        TrainingImagePath,
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='categorical')


                
                test_set = test_datagen.flow_from_directory(
                        TrainingImagePath,
                        target_size=(64, 64),
                        batch_size=32,
                        class_mode='categorical')

            
                test_set.class_indices

                '''############ Creating lookup table for all faces ############'''
                
                TrainClasses=training_set.class_indices

                
                ResultMap={}
                for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
                        ResultMap[faceValue]=faceName

               
                import pickle
                with open("ResultsMap.pkl", 'wb') as fileWriteStream:
                        pickle.dump(ResultMap, fileWriteStream)

                
                print("Mapping of Face and its ID",ResultMap)

             
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
               
                
                EndTime=time.time()

                return "Predicted face is: "+ResultMap[np.argmax(result)]+" with accuracy of " + str(predicted_accuracy) +" and time to predict of " + str(EndTime-StartTime)
                
