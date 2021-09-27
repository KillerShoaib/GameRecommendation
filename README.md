
# Game Recommendation Using Python And Scikit-Learn
This is a game recommendation system based on machine learning that provides content based recommendation.

The details of the game was collected from the [**Rawg api.**](https://rawg.io/apidocs)
You need to create an account and collect the api key to use the api.

To get all the name of games from **2000-2021** I scrape the wikipedia
table using ```beautifulsoup4``` and get the info of those game name from the api.

Live Demo : https://gamerecomendation.azurewebsites.net/
## Demo

[**Game Recommendation Website**](https://gamerecomendation.azurewebsites.net/)

![ss1](https://user-images.githubusercontent.com/59968346/134926108-5249b867-17ad-4177-99cd-e0a96f6c9398.PNG)

![ss2](https://user-images.githubusercontent.com/59968346/134926311-a6407871-610b-4602-93ef-11446f14b522.PNG)


  
## Architecture

![Overview](https://user-images.githubusercontent.com/59968346/134916794-8c42b0a2-261d-4dd1-9f31-42b085210f08.png)
## Project Steps

###  Gather Infromation Through API

* Get the game names from the wikipedia **"20xx_in_videos_games"** webpage
* Scrape the wikipedia table from that page (only the game name)
```
## for 2000-2010

table = soup.findAll("table", {"class": "wikitable sortable"})
game_names1 = table[0].findAll("i")     # findAll return whole thing as a one element list so I access that element by index
game_names2 = table[1].findAll("i")
game_names1_text = [k.get_text() for k in game_names1]
game_names2_text = [k.get_text() for k in game_names2]
if(len(game_names1_text)>len(game_names2_text)):
    game_names = game_names1_text
else:
    game_names = game_names2_text
game_list.extend(game_names)
```
```
## for 2011-2021 (only for jan-march, check the full code in the notebook)

tid = 'January\\.E2\\.80\\.93March' ## header id of that table
table_1 = soup.select_one("h3:has(span#{}) + table.wikitable".format(tid)) 
game_names_1 = table_1.select("i")  ## getting all the italic tag of that table
game_list.extend([k.get_text() for k in game_names_1])
```

* Call the Api and get Infromation
```
url = api_url+key+"&search="+i   # searching the game
response = requests.get(url)
js_format = response.json()
slug = js_format.get('results')[0].get("slug")  # getting the slug name so that we can get complete information of that game
url2 = api_game+slug+key    # requesting with slug name it will return information of that game
response_2 = requests.get(url2)
js_format2 = response_2.json()
```
  
### Create The Model

* Clean the data
* Change the format of the data from string to object
* create a soup and pass it to **counvectorizer**
* Build the model using **cosine similarity**
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count_vec = CountVectorizer(stop_words="english")

count_matrix = count_vec.fit_transform(df_copy["Soup_with_dis"])
cos_sim_dis_soup = cosine_similarity(count_matrix)
cos_sim_dis_soup
```

* Save the Model using gzip so that the size of the file is less than 100mb
```
with gzip.open(file_path,"wb") as f:
    pickled = pickle.dumps(cos_sim_dis_soup)
    optimized_pickle = pickletools.optimize(pickled)
    f.write(optimized_pickle)
```
### Create a backend using **Flask**
* import all the neccesary libraries
* initialize the flask app
```
app = Flask(__name__,template_folder='Templates')
```
* Create a function to return the top 10 similar game from the model
```
def game_rec(name):
    try:
        idx = indices[name]
        similar_score = model[idx]
        similar_game = list(enumerate(similar_score))
        sort_game = sorted(similar_game,key= lambda x:x[1], reverse=True)
        sort_game_10 = sort_game[1:11]
        game_indices = [i[0] for i in sort_game_10]
        values = information(game_indices)  # function to get the information from the api
        return values
    except:
        return None
```
* Create a route and call the function above and pass the value to the template

```
@app.route("/predict",methods=["POST"])
def predict():
    user_game_name = clean(request.form.get("name"))  # applying the clean function to filter some user result to match the game_name
    rec_game_list = game_rec(user_game_name)  # the return will be a list or none value

    if(rec_game_list!=None):
        return render_template("Predict.html",rec_game_list=rec_game_list)
    else:
        return render_template("Not_found.html")
```

### Deploy on Microsoft Azure
* Create a **requirements.txt** using pip freeze comand int the virtual env
```
pip freeze>requirements.txt
```
* Push the project on **GitHub** repository
* Create a resource group 
* Create an app service in the portal and link the GitHub repository and here you go the website will be live after the deployment is complete.


## How Cosine Similarity Work

**Cosine similarity** is a measure of similarity between two non-zero vectors of an inner product space.
It doesn't rely on the size of the vector. Rather the similarity is measured by the angular distance.
If the angular distance between two vector is less, than the vector has a higher similarity score.
And the opposite is also true. It is different than **K Nearest Neighbor**. KNN is dependent on euclidean distance
and cosine similarity is dependent on angular distance. So 2 different text which have much similar content but the size
of the content differ a lot. In this case cosine similarity can give much better result because of angular distance.

![soft-cosine](https://user-images.githubusercontent.com/59968346/134943623-daf129ac-5c20-479a-a56b-d603e79bd9f3.png)
  
* For Furhter Reading : [Understanding Cosine Similarity And Its Application](https://towardsdatascience.com/understanding-cosine-similarity-and-its-application-fd42f585296a)
## Acknowledgements

 - [Krish Naik's Movie Recommendation Video](https://www.youtube.com/watch?v=8KO-rdsWMjk&t=5s)
 - [Kaggle Notebok- Getting Started with a Movie Recommendation](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system)
 - [Kishan's Movie-Recommendation-with-Sentiment-Analysis](https://github.com/kishan0725/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis)

