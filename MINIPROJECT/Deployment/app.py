# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:15:00 2020

@author: MANTHRI BHARADWAJ
"""
from flask import Flask,render_template,request,redirect
import dash
import dash_core_components as dcc
import dash_html_components as html
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/Data')
def data():
	#return '<a href="http://127.0.0.1:8050/" style="color:red;">HI, Please click ME TO VERIFY THAT YOU ARE HUMAN</a>'
    return redirect('http://127.0.0.1:8050/',code=301)

if __name__ == '__main__':
    app.run(debug=False)