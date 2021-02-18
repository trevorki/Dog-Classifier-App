import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
import altair as alt
# import os
import base64
import numpy as np
import pandas as pd
import json
import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], title="DogStory")
server = app.server

############# Global variables ############
image_directory = 'img/breed_samples'
imagenet_labels = json.load(open("src/imagenet_class_index.json"))
imagenet_labels = [imagenet_labels[str(k)][1] for k in range(len(imagenet_labels))]
imagenet_labels= [imagenet_labels[i].capitalize() for i in range(len(imagenet_labels))]

############# Neural network setup #############
densenet = models.densenet121(pretrained=True)
densenet.eval();                        # go into evaluation mode (deactivate dropout, batchnorm, etc)
for param in densenet.parameters():     # Freeze parameters so we don't update them
    param.requires_grad = False

############## Dashboard components ##############

upload_image_card = dbc.Card(
    [
        dbc.CardHeader("Upload an image of a dog:"),
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag/Drop or Click here']),
                style = {'width': '96%', 'height': '80px', 'lineHeight': '80px', "background-color": "#fff9e8", 'borderWidth': '1px', 'borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '5px'},),
        # html.P(children = "- or -", style={"textAlign":"center"}),
        # dbc.Button("Pick a random dog", id="random-button", className="mr-2"),
        # html.Br(),
        html.Br(),
        html.P(children = "Current dog:", style={"textAlign":"left"}),
        html.Br(),
        html.Div(id='output-uploaded-image')])    # image of uploaded image
    ],
    style={"border-width": "1","width": "100%","height": "550px", "textAlign":"center"})


prediction_card = dbc.Card([
    dbc.CardHeader("Predictions:"),
    html.Iframe(id="prediction-plot", style={"border-width": "0","width": "100%","height": "375px"}),
    html.P(children = "These predictions reflect how confident the model is of the prediction. Sometimes other items in the image might be predicted.")],
    style={"border-width": "1","width": "100%","height": "550px", "textAlign":"center"}),
collapse = html.Div(
    dbc.Button(
        "Learn more", 
        id="collapse-button", 
        className="mb-3", 
        outline=False, style={"margin-top": "10px", "width": "150px", "background-color": "#c3e6e8", "color": "black"}))
photo_display_card = dbc.Card(
    [
        dbc.CardHeader("Top 3 Dog Predictions:"),
        html.Div(id ='photo-1'),
        html.Div(id='link-1'),
        html.Br(),
        html.Div(id='photo-2'),
        html.Div(id='link-2'),
        html.Br(),
        html.Div(id='photo-3'),
        html.Div(id='link-3'),
    ],
    style={"border-width": "1","width": "100%","height": "550px", "textAlign":"center"})

############## Dashboard layout ###########################

jumbotron = dbc.Jumbotron(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "DogStory - What's my breed?",
                            className="display-4",
                            style={"text-align": "left", "font-size": "40px", "color": "black",}),
                        dbc.Collapse(
                            html.P(
                                "Upload a photo of your dog and a machine learning algorithm will predict what the breed is. This doesn't just work for dogs, it will recognise 1000 different animals and objects.",
                                className="lead",
                                style={"width": "100%"},),
                            id="collapse",)], 
                    md=10),
                dbc.Col([collapse]),])),
    fluid=True,
    style={"padding": 5, "background-color": "#d1fff7"},#7cfcef
)
info_area = dbc.Container(
    dbc.Row([
            dbc.Col(upload_image_card, md=3),
            dbc.Col(prediction_card, md=6),
            dbc.Col(photo_display_card, md=3)
            ]))

app.layout = html.Div([jumbotron, info_area])
############## functions#######################

def get_predictions(img):
    k = 10  # number of predictions to return
    image_tensor = transforms.functional.to_tensor(img.resize((224, 224))).unsqueeze(0)  # convert image to tensor
    scores, label_ids = torch.softmax(densenet(image_tensor), dim=1).topk(k)           # get prediction score and label indeces for top_n results
    k_labels = [imagenet_labels[label_id.item()] for label_id in label_ids[0]]
    k_breeds = [k_labels[i].replace("_", " ") for i in range(len(k_labels))]
    k_scores = scores[0].tolist()
    k_scores = [round(k_scores[i]*100, 1) for i in range(len(k_labels))]
    predictions = pd.DataFrame({"k_label" : k_labels, "breed" : k_breeds,'score' : k_scores})
    return predictions


def prepare_photo(contents, width, height):
    div = html.Div(
        html.Img(src=contents, id='photo', style={'border-width': '0',  'width': width, 'height': height}))
    return div

def prepare_pred_image(breed_list, pred_list, index, width, height):
    if index >= len(breed_list):
        photo_source = image_directory + "/" + "Not_a_dog.jpg"
        text = "Not a dog"
        # page = wiki_wiki.page("Rick Astley")
            
        link = "https://en.wikipedia.org/wiki/" 
    else:
        photo_source = image_directory + "/" + pred_list[index] + ".jpg"
        text = f"#{index+1}: {breed_list[index]}"
        # page = wiki_wiki.page(breed_list[index])
        link = "https://en.wikipedia.org/wiki/" + breed_list[index]
    photo_base64 = "data:image/jpeg;base64,"+base64.b64encode(open(photo_source, 'rb').read()).decode('ascii')
    pred = prepare_photo(photo_base64, width, height)
    link = dcc.Link(text, href = link, target="_blank")
    return pred, text, link

@app.callback(
    dash.dependencies.Output('output-uploaded-image', 'children'),
    dash.dependencies.Input('upload-image', 'contents'))
def update_photo(contents):
    if contents is not None:
        children = prepare_photo(contents,'90%', '')
        return children

# @app.callback(
#     Output("upload-image", "contents"), 
#     # Input('upload-image', 'contents')
#     Input("random-button", "n_clicks"))
# def on_button_click(n):
#     random_photo = random.choice(os.listdir(image_directory))
#     print(random_photo)
#     if n is None:
#         return "Not clicked."
#     else:
#         return f"Clicked {n} times."
#     return
    

@app.callback(
    dash.dependencies.Output('prediction-plot', 'srcDoc'),
    dash.dependencies.Output('photo-1', 'children'), 
    dash.dependencies.Output('photo-2', 'children'),
    dash.dependencies.Output('photo-3', 'children'),
    dash.dependencies.Output('link-1', 'children'), 
    dash.dependencies.Output('link-2', 'children'), 
    dash.dependencies.Output('link-3', 'children'), 
    dash.dependencies.Input('upload-image', 'contents'))
def update_predictions(contents):
    if contents is not None:
        if len(contents.split(","))==2:
            im_bytes = base64.b64decode(contents.split(",")[1])   # seperate prefix from file infoa and convert to binary image
        else:
            im_bytes = base64.b64decode(contents)
        im_file = BytesIO(im_bytes)                           # convert binary image to file-like object
        img = Image.open(im_file)
        predictions = get_predictions(img)                   # pass to CNN and get reulting dataframe

        width = ''
        height = '120px'
        
        breed_list = predictions['breed'].tolist()
        pred_list = predictions['k_label'].tolist()         # used to show photos of the top 3 predictions
        
        ind = []                                            # filter out predictions that are not dogs
        for i in range(len(pred_list)):
            if pred_list[i] not in imagenet_labels[151:269]: # imagenet_labels[151:269] are the dogs
                ind.append(i)
        for index in sorted(ind, reverse=True):
            del pred_list[index]
            del breed_list[index]
    
        pred_1, text_1, link_1= prepare_pred_image(breed_list, pred_list, 0, width, height)
        pred_2, text_2, link_2 = prepare_pred_image(breed_list, pred_list, 1, width, height)
        pred_3, text_3, link_3 = prepare_pred_image(breed_list, pred_list, 2, width, height)

        chart = (alt.Chart(predictions)
        .mark_bar(size=20)
        .encode(
            x = alt.X("score", title = "Prediction certainty (%)"), 
            y = alt.Y("breed", sort = '-x', title = ""), 
            tooltip = ["breed", "score"])
            .configure_axis(labelFontSize=14, titleFontSize=18)
            .properties(width=300, height=250))  

        return [chart.to_html(), pred_1, pred_2, pred_3, link_1, link_2, link_3, ]


@app.callback(
    dash.dependencies.Output("collapse", "is_open"),
    [dash.dependencies.Input("collapse-button", "n_clicks")],
    [dash.dependencies.State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=False)