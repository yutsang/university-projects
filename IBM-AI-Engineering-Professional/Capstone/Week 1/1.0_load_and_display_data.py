#!/usr/bin/env python
# coding: utf-8

# <a href="http://cocl.us/pytorch_link_top">
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/Pytochtop.png" width="750" alt="IBM Product " />
# </a> 
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/cc-logo-square.png" width="200" alt="cognitiveclass.ai logo" />
# 

# <h2>Objective</h2><ul><li> How to download and pre-process the Concrete dataset.</li></ul> <p>Crack detection has vital importance for structural health monitoring and inspection. We would like to train a network to detect Cracks, we will denote the images that contain cracks as positive and images with no cracks as negative. In this lab you are going to have to download the data and study the dataset. There are two questions in this lab, including listing the path to some of the image files as well as plotting a few images. Remember the results as you will be quizzed on them. </p>
# 

# <h2>Table of Contents</h2>
# 

# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <ul>
#     <li><a href="#download_data"> Download data</a></li>
#     <li><a href="#auxiliary"> Imports and Auxiliary Functions </a></li>
#     <li><a href="#examine_files">Examine Files</a></li>
#     <li><a href="#Question_1">Question 1 </a></li>
#     <li><a href="#Display">Display and Analyze Image With No Cracks    </a></li>
#     <li><a href="#Question_2">Question 2 </a></li>
# </ul>
# <p>Estimated Time Needed: <strong>25 min</strong></p>
#  </div>
# <hr>
# 

# <h2 id="download_data">Download Data</h2>
# 

# In this section, you are going to download the data from IBM object storage using <b>wget</b>, then unzip them.  <b>wget</b> is a command the retrieves content from web servers, in this case its a zip file. Locally we store the data in the directory  <b>/resources/data</b> . The <b>-p</b> creates the entire directory tree up to the given directory.
# 

# First, we download the file that contains the images:
# 

# In[1]:


get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip -P /resources/data')


# We then unzip the file, this ma take a while:
# 

# In[2]:


get_ipython().system('unzip -q  /resources/data/concrete_crack_images_for_classification.zip -d  /resources/data')


# We then download the files that contain the negative images:
# 

# <h2 id="auxiliary">Imports and Auxiliary Functions</h2>
# 

# The following are the libraries we are going to use for this lab:
# 

# In[3]:


from PIL import Image
from matplotlib.pyplot import imshow
import pandas
import matplotlib.pylab as plt
import os
import glob


# We will use this function in the lab to plot:
# 

# In[4]:


def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])


# <h2 id="examine_files">Examine Files </h2>
# 

# In this section we are going to get a list of the negative image files, then plot them. Then for the first question your job to do something similar to the positive files. 
# 

# The path to all the images are stored in the variable  <code>directory</code>. 
# 

# In[5]:


directory="/resources/data"


# The images with out the cracks are stored in the file <b>Negative</b>
# 

# In[6]:


negative='Negative'


# We can find the path to the file with all the negative images by  using the function <code>os.path.join</code>. Inputs are the variable directory as well as the variable  <code>negative</code>.
# 

# In[7]:


negative_file_path=os.path.join(directory,negative)
negative_file_path


# <h3> Loading the File Path of Each Image </h3>
# 

# We need each the path of each image, we can find the all the file in the directory  <code>negative_file_path</code> using the function <code>os.listdir</code>, the result is a list. We print out the first three elements of the list.
# 

# In[8]:


os.listdir(negative_file_path)[0:3]


# We need the full path of the image so we join them as above. Here are a few samples  three samples:
# 

# In[9]:


[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path)][0:3]


# In some cases, we may have files of a different type, so we have to ensure it's of type <b>jpg</b>. We have to check the extension using the method <code> endswith()</code>. The method  <code>endswith()</code> returns True if the string ends with the specified suffix, otherwise, it will return False. Let's do a quick example: 
# 

# In[10]:


print("test.jpg".endswith(".jpg"))
print("test.mpg".endswith(".jpg"))


# We now have all the tools to create a list with the path to each image file.  We use a List Comprehensions  to make the code more compact. We assign it to the variable <code>negative_files<code> , sort it in and display the first three elements:
# 

# In[11]:


negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
negative_files[0:3]


# <h2 id="Question_1">Question 1</h2>
# 

# <b>Using the procedure above, load all the images with cracks paths into a list called positive files, the directory of these images is called Positive.  Make sure the list is sorted and display the first three elements of the list you will need this for the question so remember it.</b>
# 

# In[16]:


positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]


# In[17]:


positive_files.sort()


# In[18]:


positive_files[0:3]


# <h2 id="Display">Display and Analyze Image With No Cracks</h2>
# 

# We can open an image by using the <code>Image</code> Module in the  <b>PIL</b> library, using the function open. We only require the image path; the input is the path of the image. For example we can load the first image as follows:
# 

# In[13]:


image1 = Image.open(negative_files[0])
# you can view the image directly 
#image 


# we can plot the image
# 

# In[14]:


plt.imshow(image1)
plt.title("1st Image With No Cracks")
plt.show()


# We can also plot the second image.
# 

# In[15]:


image2 = Image.open(negative_files[1])
plt.imshow(image2)
plt.title("2nd Image With No Cracks")
plt.show()


# <h2 id="Question_2">Question 2</h2>
# 

# <b>Plot the first three images for the dataset with cracks. Don't forget. You will be asked in the quiz, so remember the image. </b>
# 

# In[19]:


for i in range(3):
    image= Image.open(positive_files[i])
    plt.imshow(image)
    plt.title("{} Image With Crack".format(i+1))
    plt.show()


# <hr>
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

#  [Alex Aklson](https://www.linkedin.com/in/aklson?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork-20647850&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork-20647850&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ). Ph.D., is a data scientist in the Digital Business Group at IBM Canada. Alex has been intensively involved in many exciting data science projects such as designing a smart system that could detect the onset of dementia in older adults using longitudinal trajectories of walking speed and home activity. Before joining IBM, Alex worked as a data scientist at Datascope Analytics, a data science consulting firm in Chicago, IL, where he designed solutions and products using a human-centred, data-driven approach. Alex received his Ph.D. in Biomedical Engineering from the University of Toronto.
# 

# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
# | ----------------- | ------- | ---------- | ----------------------------------------------------------- |
# | 2020-09-18        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

# In[ ]:




