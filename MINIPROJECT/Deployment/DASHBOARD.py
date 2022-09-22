import dash
import dash_core_components as dcc
import dash_html_components as html
import requests
from bs4 import BeautifulSoup
import tweepy
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import tweepy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn. preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import numpy as np
from flask import Flask,render_template,request,redirect
def Sensex_data_analysis():
	source = pd.read_html("https://www.moneyworks4me.com/best-index/bse-stocks/top-bse30-companies-list/")
	s1= source[0]
	for i in range(len(s1)):
		if s1["Company Name (M.Cap)"][i].index(" Add to Add to Watchlist Add to Boughtlist")>=0:
			a=s1["Company Name (M.Cap)"][i].index(" Add to Add to Watchlist Add to Boughtlist")
			b=s1["Company Name (M.Cap)"][i]
			s1["Company Name (M.Cap)"][i]=b[0:a]
	fig = go.Figure(go.Bar(            x=s1["Company Name (M.Cap)"],
             y=s1["Market Cap (Cr)"],
             orientation='v'))
    
	fig.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
    font=dict(
            color="White"
        ),
    paper_bgcolor="#161A27",
	)
	roe=s1.nlargest(5,'ROE')
	pe=s1.nlargest(5,'P/E')
	pbv=s1.nlargest(5,'P/BV')
	return fig,roe,pe,pbv

def Nifty_data_analysis():
	source = pd.read_html("https://www.moneyworks4me.com/best-index/nse-stocks/top-nifty50-companies-list/")
	s1= source[0]
	fig = go.Figure(go.Bar(
            x=s1["Company Name (M.Cap)"],
            y=s1["Market Cap (Cr)"],
            orientation='v'))
	fig.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
    font=dict(
            color="White"
        ),
    paper_bgcolor="#161A27",
	)
	roe=s1.nlargest(5,'ROE')
	pe=s1.nlargest(5,'P/E')
	pbv=s1.nlargest(5,'P/BV')
	return fig,roe,pe,pbv

consumerKey = "Bsvpii0oqpIvD08GunMtQ648U"
consumerSecret = "kU3s9x9T0pb8Ke5S0Rql8KOQILqQfEJbDgKvegSrQjC0dceuKZ"
accessToken = "905446212416880641-3fxMK8WC8848U2TR5UA0PA5V3F5aBuY"
accessTokenSecret = "ZCoJvv7XrSNRQLSVXGMldjUtrgEDzH9A7ONzvlI2GKJO0"

def tweetstalk():
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)
    username = "INDMarketsLIVE"
    tweets = []
    present=datetime.today()
    since=since = datetime.today() - timedelta(days=1)
    tmpTweets = api.user_timeline(username)
    for tweet in tmpTweets:
        if tweet.created_at < present and tweet.created_at > since:
            tweets.append(tweet)
            while (tmpTweets[-1].created_at > since):
                tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
                for tweet in tmpTweets:
                    if tweet.created_at < present and tweet.created_at > since:
                        tweets.append(tweet)
    username = "BT_India"
    tmpTweets = api.user_timeline(username)
    for tweet in tmpTweets:
        if tweet.created_at < present and tweet.created_at > since:
            tweets.append(tweet)
            while (tmpTweets[-1].created_at > since):
                tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
                for tweet in tmpTweets:
                    if tweet.created_at < present and tweet.created_at > since:
                        tweets.append(tweet)
    l1=[]
    for tweet in tweets:
        l1.append(tweet.text)
    return l1
def newsfordash():
    url = 'https://www.moneycontrol.com/news/tags/sensex.html'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'lxml')
    s=[]
    p1=[]
    for heading in soup.find_all(["span","h2"]):
        if heading.name=="span":
            if "IST" in heading.text.strip():
                s.append(heading.text.strip())
        else:
            p1.append(heading.text.strip())    
    p1=p1[0:len(p1)-(len(p1)-len(s))]
    url = 'https://www.moneycontrol.com/news/tags/nifty.html'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'lxml')
    for heading in soup.find_all(["span","h2"]):
        if heading.name=="span":
            if "IST" in heading.text.strip():
                s.append(heading.text.strip())
        else:
            p1.append(heading.text.strip())    
    p1=p1[0:len(p1)-(len(p1)-len(s))]
    return p1
def sensexclose():
    present=datetime.today()
    span=180
    data=[]
    while(len(data)<=180):
        since = datetime.today() - timedelta(days=span)
        data = yf.download("%5ENSEI", start=since, end=present)
        span=span+1
        data.reset_index(level=0, inplace=True)
    fig = go.Figure(go.Bar(
             x=data["Date"],
             y=data['Close'],
             orientation='v'))
    fig.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
        font=dict(
            color="White"
        ),
        paper_bgcolor="#161A27",
	)
    return fig
def niftyclose():
    present=datetime.today()
    span=180
    data=[]
    while(len(data)<=180):
        since = datetime.today() - timedelta(days=span)
        data = yf.download("%5EBSESN", start=since, end=present)
        span=span+1
        data.reset_index(level=0, inplace=True)
    fig = go.Figure(go.Bar(
             x=data["Date"],
             y=data['Close'],
             orientation='v'))
    fig.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
    font=dict(
            color="White"
        ),
    paper_bgcolor="#161A27",
	)
    return fig
bc=sensexclose()
bd=niftyclose()
def datagraber():
    def Nifty():
        present=datetime.today()
        span=10
        data=[]
        while(len(data)<=10):
            since = datetime.today() - timedelta(days=span)
            data = yf.download("%5ENSEI", start=since, end=present)
            span=span+1
        data.reset_index(level=0, inplace=True)
        for i in range(0,len(data)):
            data['Date'][i]=data['Date'][i].strftime("%Y-%m-%d")
        url = 'https://www.moneycontrol.com/news/tags/nifty.html'
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'lxml')
        date=[]
        news=[]
        for heading in soup.find_all(["span","p"]):
            if heading.name=="span":
                if "IST" in heading.text.strip():
                    date.append(heading.text.strip())
                else:
                    news.append(heading.text.strip())     
        news=news[0:len(news)-(len(news)-len(date))]
        next_page=""
        for i in range(2,15):
            next_page=url+"/page-"+str(i)+"/"
            reqs = requests.get(next_page)
            soup = BeautifulSoup(reqs.text, 'lxml')
            for heading in soup.find_all(["span","p"]):
                if heading.name=="span":
                    if "IST" in heading.text.strip():
                        date.append(heading.text.strip())
                else:
                    news.append(heading.text.strip())     
            news=news[0:len(news)-(len(news)-len(date))]
        for i in range(len(date)):
            datetime_obj = datetime.strptime(date[i], '%b %d, %Y %H:%M %p IST')
            a=datetime_obj.date()
            date[i]=a.strftime("%Y-%m-%d")
        d1=since.strftime("%Y-%m-%d")
        s1=set(date)
        l1=[]
        l2=[]
        for i in s1:
            l1.append(i)
            a=date.index(i)
            b=date.count(i)
            l=[]
            for j in range(a,a+b):
                l.append(news[j])
            l2.append(l)
        mapped=list(zip(l1,l2))
        df = pd.DataFrame(mapped, columns = ['Date', 'News']) 
        df = df.sort_values(by="Date")
        k=[]
        for i in df['News']:
            i=str(i)
            i=i.replace('"',"'")
            li = list(i.split("',"))
            li[len(li)-1]=li[len(li)-1][:len(li[len(li)-1])-2]
            a=""
            for j in li:
                j=j[2:]
                a=a+j
            tb=TextBlob(a)
            k.append(tb.sentiment.polarity)
        df['NewsSentiment']=k
        
        # credentials from https://apps.twitter.com/
        consumerKey = "Bsvpii0oqpIvD08GunMtQ648U"
        consumerSecret = "kU3s9x9T0pb8Ke5S0Rql8KOQILqQfEJbDgKvegSrQjC0dceuKZ"
        accessToken = "905446212416880641-3fxMK8WC8848U2TR5UA0PA5V3F5aBuY"
        accessTokenSecret = "ZCoJvv7XrSNRQLSVXGMldjUtrgEDzH9A7ONzvlI2GKJO0"


        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)

        api = tweepy.API(auth)

        username = "INDMarketsLIVE"
    
        tweets = []
        tmpTweets = api.user_timeline(username)
        for tweet in tmpTweets:
            if tweet.created_at < present and tweet.created_at > since:
                tweets.append(tweet)

        while (tmpTweets[-1].created_at > since):
            #print("Last Tweet @", tmpTweets[-1].created_at, " - fetching some more")
            tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
            for tweet in tmpTweets:
                if tweet.created_at < present and tweet.created_at > since:
                    tweets.append(tweet)

        date=[]
        tweet123=[]
        for tweet in tweets:
            date.append((tweet.created_at).strftime("%Y-%m-%d"))
            tweet123.append(tweet.text)
        d1=set(date)
        l11=[]
        l22=[]
        for i in d1:
            l11.append(i)
            a=date.index(i)
            b=date.count(i)
            ld=[]
            for j in range(a,a+b):
                ld.append(tweet123[j])
            l22.append(ld)
        mapped=zip(l11,l22)
        k=[]
        df1=pd.DataFrame(mapped, columns = ['Date','Tweets'])
        for i in df1['Tweets']:
            i=str(i)
            i=i.replace('"',"'")
            li = list(i.split("',"))
            li[len(li)-1]=li[len(li)-1][:len(li[len(li)-1])-2]
            a=""
            for j in li:
                j=j[2:]
                a=a+j
            tb=TextBlob(a)
            k.append(tb.sentiment.polarity)
        df1['SocialSentiment']=k
        a=pd.merge(data, df, on='Date')
        Nifty=pd.merge(a ,df1 ,on='Date')
        df=Nifty[['Open','Close',"NewsSentiment",'SocialSentiment',"Date",'News',"Tweets","High","Low","Volume"]]
        l=[]
        for i in range(0,len(df)):
            a=df['Close'][i]-df['Open'][i]
            if a>0:
                l.append(1)
            elif a<0:
                l.append(-1)
            else:
                l.append(0)
        npArray = np.array([[df['Open'][i],df['Close'][i],df['NewsSentiment'][i],df['SocialSentiment'][i],l[i]] for i in range(len(df))])
        return npArray
        
    def Sensex():
        present=datetime.today()
        span=10
        data=[]
        while(len(data)<10):
            since = datetime.today() - timedelta(days=span)
            data = yf.download("%5EBSESN", start=since, end=present)
            span=span+1
        data.reset_index(level=0, inplace=True)
        for i in range(0,len(data)):
            data['Date'][i]=data['Date'][i].strftime("%Y-%m-%d")
        url = 'https://www.moneycontrol.com/news/tags/sensex.html'
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'lxml')
        date=[]
        news=[]
        for heading in soup.find_all(["span","p"]):
            if heading.name=="span":
                if "IST" in heading.text.strip():
                    date.append(heading.text.strip())
                else:
                    news.append(heading.text.strip())     
        news=news[0:len(news)-(len(news)-len(date))]
        next_page=""
        for i in range(2,15):
            next_page=url+"/page-"+str(i)+"/"
            reqs = requests.get(next_page)
            soup = BeautifulSoup(reqs.text, 'lxml')
            for heading in soup.find_all(["span","p"]):
                if heading.name=="span":
                    if "IST" in heading.text.strip():
                        date.append(heading.text.strip())
                else:
                    news.append(heading.text.strip())     
            news=news[0:len(news)-(len(news)-len(date))]
        for i in range(len(date)):
            datetime_obj = datetime.strptime(date[i], '%b %d, %Y %H:%M %p IST')
            a=datetime_obj.date()
            date[i]=a.strftime("%Y-%m-%d")
        d1=since.strftime("%Y-%m-%d")
        s1=set(date)
        l1=[]
        l2=[]
        for i in s1:
            l1.append(i)
            a=date.index(i)
            b=date.count(i)
            l=[]
            for j in range(a,a+b):
                l.append(news[j])
            l2.append(l)
        mapped=list(zip(l1,l2))
        df = pd.DataFrame(mapped, columns = ['Date', 'News']) 
        df = df.sort_values(by="Date")
        k=[]
        for i in df['News']:
            i=str(i)
            i=i.replace('"',"'")
            li = list(i.split("',"))
            li[len(li)-1]=li[len(li)-1][:len(li[len(li)-1])-2]
            a=""
            for j in li:
                j=j[2:]
                a=a+j
            tb=TextBlob(a)
            k.append(tb.sentiment.polarity)
        df['NewsSentiment']=k
        since = datetime.today() - timedelta(days=span+1)

        # credentials from https://apps.twitter.com/
        consumerKey = "Bsvpii0oqpIvD08GunMtQ648U"
        consumerSecret = "kU3s9x9T0pb8Ke5S0Rql8KOQILqQfEJbDgKvegSrQjC0dceuKZ"
        accessToken = "905446212416880641-3fxMK8WC8848U2TR5UA0PA5V3F5aBuY"
        accessTokenSecret = "ZCoJvv7XrSNRQLSVXGMldjUtrgEDzH9A7ONzvlI2GKJO0"


        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)

        api = tweepy.API(auth)

        username = "BT_India"
    
        tweets = []
        tmpTweets = api.user_timeline(username)
        for tweet in tmpTweets:
            if tweet.created_at < present and tweet.created_at > since:
                tweets.append(tweet)

        while (tmpTweets[-1].created_at > since):
            #print("Last Tweet @", tmpTweets[-1].created_at, " - fetching some more")
            tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
            for tweet in tmpTweets:
                if tweet.created_at < present and tweet.created_at > since:
                    tweets.append(tweet)

        date=[]
        tweet123=[]
        for tweet in tweets:
            date.append((tweet.created_at).strftime("%Y-%m-%d"))
            tweet123.append(tweet.text)
        d1=set(date)
        l11=[]
        l22=[]
        for i in d1:
            l11.append(i)
            a=date.index(i)
            b=date.count(i)
            ld=[]
            for j in range(a,a+b):
                ld.append(tweet123[j])
            l22.append(ld)
        mapped=zip(l11,l22)
        k=[]
        df1=pd.DataFrame(mapped, columns = ['Date','Tweets'])
        for i in df1['Tweets']:
            i=str(i)
            i=i.replace('"',"'")
            li = list(i.split("',"))
            li[len(li)-1]=li[len(li)-1][:len(li[len(li)-1])-2]
            a=""
            for j in li:
                j=j[2:]
                a=a+j
            tb=TextBlob(a)
            k.append(tb.sentiment.polarity)
        df1['SocialSentiment']=k
        a=pd.merge(data, df, on='Date')
        Sensex=pd.merge(a ,df1 ,on='Date')
        df=Sensex[['Open','Close',"NewsSentiment",'SocialSentiment',"Date",'News',"Tweets","High","Low","Volume"]]
        l=[]
        for i in range(0,len(df)):
            a=df['Close'][i]-df['Open'][i]
            if a>0:
                l.append(1)
            elif a<0:
                l.append(-1)
            else:
                l.append(0)
        npArray = np.array([[df['Open'][i],df['Close'][i],df['NewsSentiment'][i],df['SocialSentiment'][i],l[i]] for i in range(len(df))])
        return npArray
    def niftymodel(c):
        dataset = pd.read_csv('FinalNifty.csv')
        X = dataset.iloc[:, 2:7].values
        y = dataset.iloc[:, 7].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        rc=RandomForestClassifier(max_features='log2', n_estimators=10)
        rc.fit(X_train, y_train)
        y_pred = rc.predict(X_test)
        a=accuracy_score(y_test,y_pred)
        while a==0.76:
            dataset = pd.read_csv('FinalNifty.csv')
            X = dataset.iloc[:, 2:7].values
            y = dataset.iloc[:, 7].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            rc=RandomForestClassifier(max_features='log2', n_estimators=10)
            rc.fit(X_train, y_train)
            y_pred = rc.predict(X_test)
            a=accuracy_score(y_test,y_pred)
        d=rc.predict(c)
        return d
    def sensexmodel(c):
        dataset = pd.read_csv('FinalSensex.csv')
        X = dataset.iloc[:, 2:7].values
        y = dataset.iloc[:, 7].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        randomclassifier=RandomForestClassifier(max_features='log2', n_estimators=10)
        randomclassifier.fit(X_train, y_train)
        y_pred = randomclassifier.predict(X_test)
        a=accuracy_score(y_test,y_pred)
        while a==0.76:
            dataset = pd.read_csv('FinalSensex.csv')
            X = dataset.iloc[:, 2:7].values
            y = dataset.iloc[:, 7].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            randomclassifier=RandomForestClassifier(max_features='log2', n_estimators=10)
            randomclassifier.fit(X_train, y_train)
            y_pred = randomclassifier.predict(X_test)
            a=accuracy_score(y_test,y_pred)
        d=randomclassifier.predict(c)
        return d
    a=Nifty()
    b=Sensex()
    #print(a,b)
    sen=sensexmodel(b)
    nif=niftymodel(a)
    senlabel=[]
    for i in range(1,11):
        senlabel.append("Day "+str(i))
    niflabel=[]
    for i in range(1,11):
         niflabel.append("Day "+str(i))
    fig = go.Figure(go.Bar(
             x=senlabel,
             y=sen,
             orientation='v'))
    fig.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
    font=dict(
            color="White"
        ),
    paper_bgcolor="#161A27",
	)
    fig1 = go.Figure(go.Bar(
             x=niflabel,
             y=nif,
             orientation='v'))
    fig1.update_layout(
     	autosize=False,
     	width=850,
     	height=450,
     	margin=dict(
         l=50,
         r=50,
         b=125,
         t=25,
    ),
    font=dict(
            color="White"
        ),
    paper_bgcolor="#161A27",
	)
    #print(len(sen),senlabel)
    return fig,fig1
def summary():
    url = 'https://in.finance.yahoo.com/quote/%5EBSESN/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'lxml')
    s=[]
    for heading in soup.find_all(["li"]):
        s.append(heading.text.strip())
    sensex= [i for i in s if "BSE" in i]
    Nifty= [i for i in s if "Nifty" in i]
    d=sensex[0].split()[1].split('X')[1].split("(")[0]
    if "+" in d:
        e=d.split('+')[0]
    else:
        e=d.split('-')[0]
    s=""
    s+=e+" ("+sensex[0].split("(")[1]
    de=Nifty[0].split()[1]
    if "+" in d:
        e=de.split('+')[0]
    else:
        e=de.split('-')[0]
    s1=""
    s1+=e+" ("+Nifty[0].split("(")[1]
    a1=sensex[0].split("(")[1]
    b1=Nifty[0].split("(")[1]
    a1=float(a1[0:len(a1)-2])
    b1=float(b1[0:len(b1)-2])
    if a1<b1:
        s2="NIFTY"
    else:
        s2="SENSEX"
    return s,s1,s2

s1,s2,s3=summary()   
a1,b1=datagraber()
fig123,sroe,spe,spbv=Sensex_data_analysis()
fig124,nroe,npe,npbv=Nifty_data_analysis()

t=tweetstalk()
trends=newsfordash()
external_stylesheets = [
    {
        'href': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css',
        'rel': 'stylesheet'
    },
    {
        'href':'https://github.com/manthribharadwaj12/test/blob/main/hi.css',
        'rel':'stylesheet'
    }
]

external_scripts = [
    {'src': 'https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js'},
    {
        'src': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js'
    }
]
app1 = dash.Dash(__name__,external_scripts=external_scripts,external_stylesheets=external_stylesheets)

app1.layout = html.Div(id='', className='',style={}, children=[
	html.Nav(id='', className='navbar navbar-inverse visible-xs',style={}, children=[
		html.Div(id='', className='container-fluid',style={}, children=[
			html.Div(id='', className='navbar-header',style={}, children=[
				html.Button(id='', className='navbar-toggle',style={}, children=[
					html.Span(id='', className='icon-bar',style={}, children=[
					]),
					html.Span(id='', className='icon-bar',style={}, children=[
					]),
					html.Span(id='', className='icon-bar',style={}, children=[
				    ]),
				]),
				html.A(id='', className='navbar-brand', href='#', style={}, children=[
				]),
			]),
			html.Div(id='myNavbar', className='collapse navbar-collapse',style={}, children=[
				html.Ul(id='', className='nav navbar-nav',style={}, children=[
					html.Li(id='', className='active',style={}, children=[
						html.A(id='', className='', href='#', style={}, children=[ 'Dashboard'
						]),
					]),
					html.Li(id='', className='',style={}, children=[
					html.A(id='', className='', href='#', style={}, children=[
						]),
					]),
					html.Li(id='', className='',style={}, children=[
						html.A(id='', className='', href='#', style={}, children=[
						]),
					]),
					html.Li(id='', className='',style={}, children=[
						html.A(id='', className='', href='#', style={}, children=[
						]),
					]),
				]),
			]),
		]),
	]),
	html.Div(id='', className='container-fluid',style={'background-color': '#1E212F','color':'white'}, children=[
		html.Div(id='', className='row content',style={'background-color': '#1E212F', 'padding-top':25}, children=[
			html.Div(id='', className='col-sm-3 sidenav hidden-xs',style={'background-color': '#1E212F'}, children=[
				html.Ul(id='', className='nav nav-pills nav-stacked',style={}, children=[
					html.Li(id='', className='active',style={}, children=[
						html.A(id='', className='', href='', style={'color':'white'}, children="Dashboard"),
					]),
					html.Li(id='', className='',style={}, children=[
						html.H4(id='', className='',style={'padding-top':25,'color':'white'}, children='Welcome to our Dashboard'),
					]),
					html.P(id='', className='',style={}, children='This helps you to analyze the stock market completely'),
				]),
			]),
			html.Div(id='', className='col-sm-9',style={'color':'white'}, children=[
				html.Div(id='', className='well',style={'background-color': '#161A27','color':'white'}, children=[
					html.Center(id='', className='',style={}, children=[
						html.H1(children='DASHBOARD FOR STOCK MARKET ANALYSIS',id='', className='',style={'color':'white'}),
					]),
				]),
				html.Div(id='', className='row',style={}, children=[
					html.Div(id='', className='col-sm-3',style={}, children=[
						html.Div(id='', className='well',style={'height':100,'background-color': '#161A27'}, children=[
							html.H4(id='', className='',style={}, children="SENSEX"+'\n'+"VS"+'\n'+"NIFTY"),
						]),
					]),
					html.Div(id='', className='col-sm-3',style={}, children=[
						html.Div(id='', className='well',style={'height':100,'background-color': '#161A27'}, children=[
							html.H4(id='', className='',style={}, children="SENSEX"+'\n'+s1),
						]),
					]),
					html.Div(id='', className='col-sm-3',style={}, children=[
						html.Div(id='', className='well',style={'height':100,'background-color': '#161A27'}, children=[
							html.H4(id='', className='',style={}, children="NIFTY "+'\n'+s2),
						]),
					]),
					html.Div(id='', className='col-sm-3',style={}, children=[
						html.Div(id='', className='well',style={'height':100,'background-color': '#161A27'}, children=[
							html.H4(id='', className='',style={}, children="OUR CHOICE FOR TODAY"),
							html.H6(id='', className='',style={}, children=s3),
						]),
					]),
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
				html.H4(children="SENSEX PREDICTION FOR NEXT 10 DAYS"),
					dcc.Graph(
        				id='sensex-graph2',
        				figure=a1,
						style={'height':400,'padding':50})
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
					html.H4(children="NIFTY PREDICTION FOR NEXT 10 DAYS"),
					dcc.Graph(
        				id='Nifty-graph2',
        				figure=b1,
						style={'height':400,'padding':50})
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
					html.H4(children="SENSEX PAST 6 MONTHS CLOSING PRICES"),
					dcc.Graph(
        				id='sensex-graph3',
        				figure=bc,
						style={'height':400,'padding':50})
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
				html.H4(children="NIFTY PAST 6 MONTHS CLOSING PRICES"),
					dcc.Graph(
        				id='nifty-graph3',
        				figure=bd,
						style={'height':400,'padding':50})
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
					html.H4(children="SENSEX COMPANIES COMPARISON (BASED ON MARKET CAPITAL)"),
					dcc.Graph(
        				id='sensex-graph1',
        				figure=fig123,
						)
				]),
				html.Div(id='', className='well',style={'padding': 100,'background-color': '#161A27'}, children=[
					html.H4(children="NIFTY COMPANIES COMPARISON (BASED ON MARKET CAPITAL)"),
					dcc.Graph(
        				id='Nifty-graph',
        				figure=fig124,)
				]),
				html.Div(id='', className='row',style={}, children=[
					html.Div(id='', className='col-sm-4',style={}, children=[
						html.Div(id='', className='well',style={'padding': 10,'background-color': '#161A27'}, children=[
							html.H5(id='', className='',style={}, children="Top 5 COMPANIES (ROE)"),
                            html.Br(),
							html.H6(id='', className='',style={}, children="SENSEX"),
                            html.Ol(id="roe1", children=[html.Li(i) for i in sroe["Company Name (M.Cap)"]]),
                            html.Br(),
							html.H6(id='', className='',style={}, children="NIFTY"),
                            html.Ol(id="roe2", children=[html.Li(i) for i in nroe["Company Name (M.Cap)"]])
						]),
					]),
					html.Div(id='', className='col-sm-4',style={}, children=[
						html.Div(id='', className='well',style={'padding': 10,'background-color': '#161A27'}, children=[
							html.H5(id='', className='',style={}, children="Top 5 COMPANIES (P/E)"),
                            html.Br(),
							html.H6(id='', className='',style={}, children="SENSEX"),
                            html.Ol(id="pe1", children=[html.Li(i) for i in spe["Company Name (M.Cap)"]]),
                            html.Br(),
							html.H6(id='', className='',style={}, children="NIFTY"),
                            html.Ol(id="pe2", children=[html.Li(i) for i in npe["Company Name (M.Cap)"]])
						]),
					]),
					html.Div(id='', className='col-sm-4',style={}, children=[
						html.Div(id='', className='well',style={'padding': 10,'background-color': '#161A27'}, children=[
							html.H5(id='', className='',style={}, children="Top 5 COMPANIES (P/BV)"),
                            html.Br(),
							html.H6(id='', className='',style={}, children="SENSEX"),
                            html.Ol(id="pbv1", children=[html.Li(i) for i in spbv["Company Name (M.Cap)"]]),
                            html.Br(),
							html.H6(id='', className='',style={}, children="NIFTY"),
                            html.Ol(id="pbv2", children=[html.Li(i) for i in npbv["Company Name (M.Cap)"]])
						]),
					]),
				]),
				html.Div(id='', className='row',style={}, children=[
					html.Div(id='', className='col-sm-8',style={}, children=[
						html.Div(id='', className='well',style={'width':750,'height':400,'padding': 10,'background-color': '#161A27','overflow-x':'hidden'},children=[
            				html.H4(children="TRENDING NEWS"),
							html.Ul(id='my-list', children=[html.Li(i) for i in trends])
        				]),
					]),
					html.Div(id='', className='col-sm-4',style={}, children=[
						html.Div(id='', className='well',style={'height': 400,'background-color': '#161A27','overflow-x':'hidden','padding': 10}, children=[
							html.H4(children="SOCIAL MEDIA TALK"),
							html.Ul(id='my-list1', children=[html.Li(i) for i in t])
						]),
					]),
				]),
			]),
		]),
	]),
])
    

if __name__ == '__main__':
    app1.run_server(debug=False)
