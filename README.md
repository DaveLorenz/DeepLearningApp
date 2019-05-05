# DeepLearningApp

This is the code used to create the following page: http://deeplearningade.appspot.com/

This was a school project in the UVA MSBA where we first used keras to create a deep learning model. The model used for this prediction engine was a word CNN without pre-trained embeddings. The application is meant to read user-entered text explaining an adverse reaction to drugs and return a probability of the event being severe. We've incorporated it into our Word Press website: http://t6tern.wpengine.com/?page_id=296.

The steps are as follows:

1. Train a prediction model (not in this repository)

2. Run the python code "Outputting pre-trained model" after training your model to save model weights as h5 file and model framework as a json file. This way you don't have to train the model each time someone uses the application.

3. Get your Flask app running locally via the main (for local run) file using Flask. Flask allows python to communicate with html files via a "templates" folder saved in the same directory (as is it structured in this repository). Watch the videos linked below for more info. These explain the application of Flask to deep learning keras models very well.

https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=1743s (overview of keras and Flask)

https://www.youtube.com/watch?v=IIi6e5oDZ68 (interacting HTML with Flask)

4. Follow the instructions in the video below to get your Flask app running via Google Cloud. The first year and $300 are free. You'll need to upload the following: (1) main.py (2) the templates folder (3) the yaml file, (4) the h5 file with weights, (5) the json file with model framework, (6) the tsv file to ensure your matrix has the same structure as the training matrices, (7) a requirement.txt file. Note, this requirement.txt file is not loaded to this repository due to security reasons. Please contact me for more information on this. In general, you want to list the library versions you relied upon such that Google Cloud can download these to the cloud (e.g., keras==2.2.4). You'll also need to specify the gunicorn version such that Google Cloud can connect to the Python web server to download the proper libraries utilized. 

https://www.youtube.com/watch?v=RbejfDTHhhg

Happy building! Please contact me if you'd like any help debugging or have any questions.
