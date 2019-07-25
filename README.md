# DeepLearningApp

See how this RESTful API interacts with users at the 0:30 mark in the following youtube video: https://www.youtube.com/watch?v=76G3Wf91JR0 (no longer on the cloud for cost reasons).

This was a school project in the UVA MSBA where we first used keras to create a deep learning model. The model used for this prediction engine was a word CNN without pre-trained embeddings. The application is meant to read user-entered text explaining an adverse reaction to drugs and return a probability of the event being severe. We've incorporated it into a Word Press website highlighted in the link above.

The code in this repo creates the first two pages (severity entry page and return prediction page). We built off of these pages to create form entry pages, but I wanted to keep this code as simple as possible for others to learn from. 

The steps are as follows:

1. Train a prediction model (not in this repository)

2. Run the python code "Outputting pre-trained model" after training your model to save model weights as h5 file and model framework as a json file. This way you don't have to train the model each time someone uses the application.

3. Get your RESTful API running locally via the main (for local run) file using Flask. Flask allows python to communicate with html files via a "templates" folder saved in the same directory (as is it structured in this repository). Watch the videos linked below for more info. These explain the application of Flask to deep learning keras models very well.

https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=1743s (overview of keras and Flask)

https://www.youtube.com/watch?v=IIi6e5oDZ68 (interacting HTML with Flask)

Note, the python code below interacts Flask with HTML. More specifically, when the user clicks a submit button on an HTML form, POST sends the id text_entered to Flask/python using request.form.get. text_entered is an id that contains the text the user enters into HTML. This text becomes textData in python. The pre-trained loaded.model generates a prediction, and render_template loads the prediction.html page and sends this prediction to HTML as prediction. In prediction.html, this probability is referred to as {{ prediction }}. When the application is first run, prior to posted, render_template loads the first page: search_page.html.


    @app.route('/', methods=['GET','POST'])
    def predict():
    
    #whenever the predict method is called, we're going    
    #to input the user entered text into the model
    #and return a prediction
    
    if request.method=='POST':    
        textData = request.form.get('text_entered')      
        textDataArray = [textData]
        textTokenized = prepDataForDeepLearning(textDataArray)
        prediction = int((1-np.asscalar(loaded_model.predict(textTokenized)))*100)
        #return prediction in new page        
        return render_template('prediction.html', prediction=prediction)      
    else:    
        return render_template("search_page.html")  

4. Follow the instructions in the video below to get your Flask app running via Google Cloud. The first year and $300 are free. You'll need to upload the following: (1) main.py (2) the templates folder (3) the yaml file, (4) the h5 file with weights, (5) the json file with model framework, (6) the tsv file to ensure your matrix has the same structure as the training matrices, (7) the requirement.txt file. Note that you need to specify the gunicorn version such that Google Cloud can connect to the Python web server to download the proper libraries utilized. 

https://www.youtube.com/watch?v=RbejfDTHhhg

Note that you will run the following commands in the cloud to get this running: (1) cd (name of folder) (2) gcloud app deploy.

Happy building! Please contact me if you'd like any help debugging or have any questions.
