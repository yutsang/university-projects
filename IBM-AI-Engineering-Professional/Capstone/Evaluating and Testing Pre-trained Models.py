#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Peer Review Final Assignment</font></h1>

# ## Introduction
# 

# 

# In this lab, you will build an image classifier using the VGG16 pre-trained model, and you will evaluate it and compare its performance to the model we built in the last module using the ResNet50 pre-trained model. Good luck!

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item41">Download Data 
# 2. <a href="#item42">Part 1</a>
# 3. <a href="#item43">Part 2</a>  
# 4. <a href="#item44">Part 3</a>  
# 
# </font>
#     
# </div>

#    

# <a id="item41"></a>

# ## Download Data

# Use the <code>wget</code> command to download the data for this assignment from here: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip

# Use the following cells to download the data.

# In[1]:


# getting imports
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input


# In[2]:


# get data
get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip')


# In[3]:


# unzip data
get_ipython().system('unzip concrete_data_week4.zip')


# After you unzip the data, you fill find the data has already been divided into a train, validation, and test sets.

#   

# <a id="item42"></a>

# ## Part 1

# In this part, you will design a classifier using the VGG16 pre-trained model. Just like the ResNet50 model, you can import the model <code>VGG16</code> from <code>keras.applications</code>.

# You will essentially build your classifier as follows:
# 1. Import libraries, modules, and packages you will need. Make sure to import the *preprocess_input* function from <code>keras.applications.vgg16</code>.
# 2. Use a batch size of 100 images for both training and validation.
# 3. Construct an ImageDataGenerator for the training set and another one for the validation set. VGG16 was originally trained on 224 × 224 images, so make sure to address that when defining the ImageDataGenerator instances.
# 4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.
# 5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.
# 6. Fit the model on the augmented data using the ImageDataGenerators.

# Use the following cells to create your classifier.

# In[4]:


# 1. imports already done in above cells


# In[4]:


# 2. defining global constants like batch size
num_classes = 2
image_resize = 224

# either training or validation batch_size is 100
batch_size = 100


# In[5]:


# 3. Constructing ImageDataGenerator instances
data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
)

# training generator
train_generator = data_generator.flow_from_directory(
    'concrete_data_week4/train',
    target_size = (image_resize, image_resize),
    batch_size = batch_size,
    class_mode = 'categorical'
)


# In[6]:


# valication generator
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week4/valid',
    target_size = (image_resize, image_resize),
    batch_size = batch_size,
    class_mode = 'categorical'
)


# In[7]:


# 4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.
model = Sequential()

model.add(VGG16(
    include_top = False,
    pooling = 'avg',
    weights = 'imagenet'
))


# In[8]:


model.add(Dense(num_classes, activation = 'softmax'))


# In[9]:


model.layers


# In[10]:


model.layers[0].trainable = False


# In[11]:


model.summary()


# In[12]:


#5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[13]:


get_ipython().run_cell_magic('time', '', '#6. Fit the model on the augmented data using the ImageDataGenerators.\nsteps_per_epoch_training = len(train_generator)\nsteps_per_epoch_validation = len(validation_generator)\nnum_epochs = 1\n\nfit_history = model.fit_generator(\n    train_generator,\n    steps_per_epoch=steps_per_epoch_training,\n    epochs=num_epochs,\n    validation_data=validation_generator,\n    validation_steps=steps_per_epoch_validation,\n    verbose=1,\n)\n')


#    

# In[14]:


model.save('classifier_vgg16_model.h5')


# <a id="item43"></a>

# ## Part 2

# In this part, you will evaluate your deep learning models on a test data. For this part, you will need to do the following:
# 
# 1. Load your saved model that was built using the ResNet50 model. 
# 2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.
# 3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).
# 4. Print the performance of the classifier using the VGG16 pre-trained model.
# 5. Print the performance of the classifier using the ResNet pre-trained model.
# 

# Use the following cells to evaluate your models.

# In[16]:


# 1. Load your saved model that was built using the ResNet50 model. 
from keras.models import load_model
resnet = load_model('classifier_resnet_model.h5')


# In[17]:


# 2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.
data_gen = ImageDataGenerator()
test_gen = data_gen.flow_from_directory(
           'concrete_data_week4/test',
           target_size = (224, 224),
           shuffle = False
)


# In[18]:


# 3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).
vgg16 = model.evaluate_generator(test_gen)
print('VGG16')
print(vgg16)
print('Loss : ', str(vgg16[0]))
print('Accuracy : ', str(vgg16[1]))
# 4. Print the performance of the classifier using the VGG16 pre-trained model.


# In[19]:


# 5. Print the performance of the classifier using the ResNet pre-trained model.
res50 = resnet.evaluate_generator(test_gen)
print('RESNET50')
print(res50)
print('Loss : ', str(res50[0]))
print('Accuracy : ', str(res50[1]))


#    

# <a id="item44"></a>

# ## Part 3

# In this model, you will predict whether the images in the test data are images of cracked concrete or not. You will do the following:
# 
# 1. Use the **predict_generator** method to predict the class of the images in the test data, by passing the test data ImageDataGenerator instance defined in the previous part as an argument. You can learn more about the **predict_generator** method [here](https://keras.io/models/sequential/).
# 2. Report the class predictions of the first five images in the test set. You should print something list this:
# 
# <center>
#     <ul style="list-style-type:none">
#         <li>Positive</li>  
#         <li>Negative</li> 
#         <li>Positive</li>
#         <li>Positive</li>
#         <li>Negative</li>
#     </ul>
# </center>

# Use the following cells to make your predictions.

# In[20]:


import numpy as np
y_pred_vgg = model.predict_generator(test_gen)
y_pred_res = resnet.predict_generator(test_gen)

def predict(x_test) :
  for predictions in x_test:
     prediction = np.argmax(predictions)
     if (prediction == 0) :
       print('Negative')
     else :
        print('Positive') 


# In[21]:


# vgg16 predictions
print('VGG16 predictions of first 5 elements in test')
predict(y_pred_vgg[0:5])


# In[22]:


# resnet50 predictions
print('ResNet50 predictions of first 5 elements in test')
predict(y_pred_res[0:5])


# In[23]:


# true values
print('True first 5 elements in test')
predict(test_gen.next()[1][0:5])


# <hr>
# 
# Copyright &copy; 2023 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
