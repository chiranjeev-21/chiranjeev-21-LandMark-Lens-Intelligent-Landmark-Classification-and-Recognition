## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project!
In this project, we will build a pipeline to process real-world, user-supplied images and to put your model into an app.
Given an image, the app will predict the most likely locations where the image was taken.

Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.

### Why Landmark Classification

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, we will take the first steps towards addressing this problem by building a CNN-powered app to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, the app will accept any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world.

#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers) that are not covered in the class.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
    ```
       git clone https://github.com/chiranjeev-21/chiranjeev-21-LandMark-Lens-Intelligent-Landmark-Classification-and-Recognition.git
       cd chiranjeev-21-LandMark-Lens-Intelligent-Landmark-Classification-and-Recognition
    ```
    
2. Create a new conda environment with python 3.7.6:

    ```
       conda create --name cnn_project -y python=3.7.6
       conda activate cnn_project
    ```
    
    NOTE: you will have to execute `conda activate cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
    ```
       pip install jupyterlab
       jupyter lab
    ```

## Checkpoints

The Pre-Trained checkpoints can be found [here](https://drive.google.com/drive/folders/1rd3lHtKcncyhT46j0ZcIWOP8j8Juw7h0?usp=sharing)
## Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
