

import pickle
import pandas as pd
import numpy as np
import webbrowser    
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly
import plotly.express as px
import sqlite3 as sql

conn = sql.connect('Prediction.db')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = None


# In[33]:


def load_model():
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)
    global vocab
    file = open("feature.pkl", 'rb')
    vocab = pickle.load(file)


# In[34]:


# def open_browser():
#     webbrowser.open_new('https://publicservants.in')


# In[35]:


def check_review(reviewText):
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)  
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(vectorised_review)


# In[36]:


# def load_data():
#     global df
#     df = pd.read_csv('balanced_review.csv')
#     df.dropna(inplace = True)
#     df = df[df['overall'] != 3]
#     df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
#     df['Names'] = np.where(df['Positivity']==1,'Positive','Negative')
#     global labels
#     labels = df['Names'].tolist()


# In[37]:


def load_scrappeddata():
    global df
    df=pd.read_sql('SELECT * FROM Predicted', conn)
    dfn=df[df['predictedvalue']==0] 
    dfn=dfn.iloc[:6,:]
    dfp=df[df['predictedvalue']==1]
    dfp=dfp.iloc[:6,:]
    df1=pd.concat([dfp,dfn],ignore_index=True)
    global reviews
    reviews = []
    for i in range(len(df1)):
        reviews.append({'label':df1['reviews'][i],'value':i})


# In[38]:


# def predict_scrappeddata():
#     global sentiment
#     sentiment = []
#     for i in range (len(df1['reviews'])):
#         response = check_review(df1['reviews'][i])
#         if (response[0]==1):
#             sentiment.append('Positive')
#         elif (response[0] ==0 ):
#             sentiment.append('Negative')
#         else:
#             sentiment.append('Unknown')


# In[57]:


def create_app_ui():
    pie_chart=px.pie(
        data_frame=df,
        values=[df['predictedvalue'].value_counts()[1],df['predictedvalue'].value_counts()[0]],
        names=['Positive Reviews','Negative Reviews'],
        color=['Positive Reviews','Negative Reviews'],
        color_discrete_sequence=['Green','Red'],
        #title='Distribution of model prediction of scrapped data',
        width=600,                          
        height=380,                         
        hole=0.5, 
    )
    
    main_layout = html.Div(
            [
                html.Hr(),
                html.H1(id = 'Main_title', children = 'Sentiment analysis with insights',
                    style={'text-align':'center','color':'red'}),
                html.Hr(),
            dbc.Row([
                dbc.Col(
                   html.Div([
                   html.H2(children='Distriution of scrapped reviews'),
                   dcc.Graph(
                   id='pie_graph',
                   figure=pie_chart)
        ],
        style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )
            ),dbc.Col(
                
                html.Div(
            [
                html.H2(children='Etsy reviews'),
                dcc.Dropdown(
                            id = 'reviewpicker',
                            options = reviews, 
                            value=None,
                            optionHeight=70,
                            style = {'margin-bottom': '30px','min-width':'670px','padding-top':'25px'}
                            ),
                dbc.Button(
                            id="check_review", children='Submit',
                            color = 'dark',style={'margin':'0 45%','padding':'5px 15px'}
                          ),
                html.Div(id='container1',style={'padding-top':'15px'})
            ],
            style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}),
            )]),
            dbc.Row([
                dbc.Col([
                html.Div(
            [
                html.H2('Try it yourself!'),
                dcc.Textarea(
                            id = 'textarea_review',
                            placeholder = 'Enter the review here...',
                            style={'width':'650px','height':'300'}
                            ),
                html.Div(id='container2',style={'padding':'15px 15px 15px 10px'})
            ],
            style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
            )
            ]),
                dbc.Col([
                html.Div(
            [   
                html.Div([
                html.H2('Word Cloud'),
                dbc.Button("ALL Words",
                 id="allbt",
                 outline=True,
                 color="info", 
                 className="mr-1",
                 n_clicks_timestamp=0,
                 style={'padding':'10px','padding-right':'15px'}
                 ),
                dbc.Button("Positve Words",
                id="posbt",
                 outline=True,
                 color="success", 
                 className="mr-1",
                 n_clicks_timestamp=0,
                 style={'padding':'10px','padding-right':'15px'}
                 ),
                dbc.Button("Negative Words",
                id="negbt",
                outline=True, 
                color="danger",
                className="mr-1",
                n_clicks_timestamp=0,
                style={'padding':'10px','padding-right':'15px'}
                )
                ],style={'padding-left':'15px'}
                ),
                html.Div(id='container',style={'padding':'15px'})
            ],
            style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )])
            ])
            ],
        style={"height": "100vh","background-color": "#d3d3d3" , "width" : "100%"}
            )
    return main_layout


# In[40]:


@app.callback(
    Output('container','children'),
    [
        Input('allbt','n_clicks_timestamp'),
        Input('posbt','n_clicks_timestamp'),
        Input('negbt','n_clicks_timestamp'),
    ]
)
def wordcloudbutton(allbt,posbt,negbt):

    if int(allbt) > int(posbt) and int(allbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('wholeword.png'))])
    elif int(posbt) > int(allbt) and int(posbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('posword.png'))
            ])
    elif int(negbt) > int(allbt) and int(negbt) > int(posbt):
        return html.Div([
            html.Img(src=app.get_asset_url('negword.png'))
            ])
    else:
        pass


# In[41]:


@app.callback(
    Output('container2', 'children'),  
    [
    Input('textarea_review', 'value')
    ]
#     ,
#     [
#     State('textarea_review', 'value')                                    
#     ]                                   
    )                                      
def review_predict(textarea_value):         
#     print("Data Type  = ", str(type(n_clicks)))  
#     print("Value      = ", str(n_clicks))
    
#     print("Data Type  = ", str(type(textarea_value)))
#     print("Data Type  = ", str(textarea_value))
    response = check_review(textarea_value)
    #if (n_clicks > 0):              
    if (response[0] == 0 ):
        return html.Div([
            dbc.Alert("Its a negative review", color="danger")
            ])
        #result = 'Negative'
    elif (response[0] == 1 ):
        return html.Div([
            dbc.Alert("Its a positive review", color="success")
            ])
        #result = 'Positive'
    else:
        return ""
        #result = 'Unknown'

    #return result
#     else:
#         return ""


# In[42]:


# @app.callback(
#     Output('result', 'style'),  
#     [
#     Input('button_review', 'n_clicks')
#     ]
#     ,
#     [
#     State('textarea_review', 'value')                                    
#     ]                                    
#     )                                      
# def review_predict(n_clicks,textarea_value):         
#     print("Data Type  = ", str(type(n_clicks)))  
#     print("Value      = ", str(n_clicks))
    
#     print("Data Type  = ", str(type(textarea_value)))
#     print("Data Type  = ", str(textarea_value))
#     response = check_review(textarea_value)
#     if (n_clicks > 0):              
#         if (response[0] == 0 ):
#             result = {'color':'red'}
#         elif (response[0] == 1 ):
#             result = {'color':'green'}
#         else:
#             result = 'Unknown'
        
#         return result
#     else:
#         return ""


# In[43]:


@app.callback(
    Output('container1','children'),
    [
        Input('check_review','n_clicks')
    ],
    [
        State('reviewpicker','value')
    ])
def review_predict2(n_clicks,value):
    review_selected = reviews[value]['label']
    response = check_review(review_selected)
    if (n_clicks>0):
        if (response[0]==0):
            return html.Div([
                dbc.Alert("Its a negative review", color="danger")
                ])
            #result = 'Negative'
        elif (response[0]==1):
            return html.Div([
                dbc.Alert("Its a Positive review", color="success")
                ])
            #result = 'Positive'
        else:
            return ""
        #return result
    else:
        return ""


# In[44]:


# @app.callback(
#     Output('result2','style'),
#     [
#         Input('check_review','n_clicks')
#     ],
#     [
#         State('reviewpicker','value')
#     ])
# def review_predict2(n_clicks,value):
#     review_selected = reviews[value]['label']
#     response = check_review(review_selected)
#     if (n_clicks>0):
#         if (response[0]==0):
#             result = {'color':'red'}
#         elif (response[0]==1):
#             result = {'color':'green'}
#         else:
#             result = 'Unknown'
#         return result
#     else:
#         return ""


# In[58]:


def main():
    print("Start of my project")
    load_model()
    #load_data()
    load_scrappeddata()
    #predict_scrappeddata()
    project_name = 'Sentiment Analysis with Insights'
    print(project_name)
    #open_browser()
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    print("End of my Project")
    
if __name__ == '__main__':
  main()



