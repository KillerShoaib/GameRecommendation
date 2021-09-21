from flask import Flask,render_template,request
import pandas as pd
import pickle as pkl
from ast import literal_eval
import requests
import json
import gzip



app = Flask(__name__,template_folder='Templates')

# loading the dataset
df = pd.read_csv("Final_game_dataset.csv")

# loading the model

file_path = "cos_sim_discri_soup.pkl"
with gzip.open(file_path,"rb") as f:
    p = pkl.Unpickler(f)
    model = p.load()



# dropping the unnecesarry columns
df = df.drop(columns = ["Metacritic","Rating_Top","Ratings","Parent_Platforms","ESRB_Rating"])

# creating a clean function to make it easy to search the games
def clean(name):
    string = name.replace("'","").replace("-"," ").replace(":","")
    string = string.lower()
    return string



# cleaning the names but not effecting the actual name values in the df
df_2 = df[["Name"]].copy()
df_2["Name"] = df_2["Name"].apply(clean)

#creating a pandas series name as index from df_2
indices = pd.Series(df.index,index=df_2["Name"])

# getting the infromation from the csv

# changing to the desirable object
features = ["Platforms","Developers","Publishers","Genres","Tags"]

for feature in features:
    df[feature] = df[feature].apply(literal_eval)

# filling the missing values
columns = ['Released',"Clean_description"]

for column in columns:
    df[column] = df[column].fillna("")

# creating every column as a list of string
columns_list = ["Developers","Publishers","Genres","Tags"]

def get_list(obj):
    
    if isinstance(obj,list):
        
        names = [i["name"] for i in obj]
        return names
    return []  # return empty list for malfromed data


# now apply the function

for column in columns_list:
    df[column] = df[column].apply(get_list)


# for platfrom column the process is like this to convert it to a list of platfroms

def get_plat_list(obj):
    
    if isinstance(obj,list):
        
        platform = [i["platform"] for i in obj]
        
        names = [i["name"] for i in platform]
        
        return names
    return []

df['Platforms'] = df['Platforms'].apply(get_plat_list)


# changing Released column values into string
def convert_to_string(obj):
  string = str(obj)
  return string

df["Released"] = df["Released"].apply(convert_to_string)


# for api
api_url = "https://api.rawg.io/api/games?key=e431b52cb8204c66b501a35ba93bd3ca&search="


def information(list_of_values):
    # this dict will store the values
    list_of_dict = []
    for i in range(len(list_of_values)):
        # here list_of_values will be index of the dataframe, we can iterate through this list and get index of that game

        name = df["Name"].iloc[list_of_values[i]]
        platfrom = " - ".join(df["Platforms"].iloc[list_of_values[i]]) 
        genres = ""
        # this block of code is to handle missing value
        try:
            genres = df["Genres"].iloc[list_of_values[i]][0] # this is a list, only getting the first value
        except:
            genres = "None"
        
        # this block of code is to handle failed api call

        image = ''
        try:
            # getting the api req for image
            api_url = "https://api.rawg.io/api/games?key=e431b52cb8204c66b501a35ba93bd3ca&search="
            response = requests.get(api_url+name)
            js_format = response.json()
            if js_format.get('results')[0].get('background_image') != None:

                image = js_format.get('results')[0].get('background_image')
            else:
                image = 'static/no-image-found.png'
            

        except:
            image = 'static/no-image-found.png'

        
        
        dict_values = {
        'name':name,
        'platforms':platfrom,
        'description':df["Clean_description"].iloc[list_of_values[i]],
        'released':df["Released"].iloc[list_of_values[i]],
        'genres':genres,  
        'image':image
        }
        list_of_dict.append(dict_values)

    return list_of_dict

## creating e function to recomend game names
def game_rec(name):
    try:
        idx = indices[name]
        similar_score = model[idx]
        similar_game = list(enumerate(similar_score))
        sort_game = sorted(similar_game,key= lambda x:x[1], reverse=True)
        sort_game_10 = sort_game[1:11]
        game_indices = [i[0] for i in sort_game_10]
        values = information(game_indices)
        return values
    except:
        return None


@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/predict",methods=["POST"])
def predict():
    user_game_name = clean(request.form.get("name"))  # applying the clean function to filter some user result to match the game_name
    rec_game_list = game_rec(user_game_name)  # the return will be a list or none value

    if(rec_game_list!=None):
        return render_template("Predict.html",rec_game_list=rec_game_list)
    else:
        return render_template("Not_found.html")



if __name__== '__main__':
    app.run(debug=True)

