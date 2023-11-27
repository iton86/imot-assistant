# IMOT ASSISTANT
v0.0.1


## BACKGROUND
The apartments in Sofia’s city center could be a good long term investment that could:\
    a)	Preserve/increase the value of the investment\
    b)	Generate income

One of the biggest websites for property ads is imot.bg. It provides good user experience, but has two main limitations:\
    a)	The location of the ads is very generic – i.e. you can filter ads only by neighborhood = ‘Center’. 
        However, there are distinctly different areas in the city center both in terms of appeal and price range.\
    b)	There is ‘Favorite’ option that can save the interesting ads, but there is no ‘Remove’ option – i.e. the same 
        ads would appear over and over again, without the option to flag them as ‘Not interested’

The application tries to address the above two limitations by providing:\
    a) Comprehensive ETL to scrape and save the ads content on a regular basis\
    b) Customized Transformer based NER (Name Entity Recognition) ML.\
    c) Interactive UI (although really basic, it serves the main purpose) 

## ARCHITECTURE
The application is logically divided into 5 main components\
    a)	imot_scraper.py - ETL script\
    b)	ImotNer.py class - ML model\
    c)	app.py - UI\
    d)	main.py - Driver script\
    e)	settings.py/imot_config.py configuration files. 

### Setup
1.	The fine-tuned NER model is too big to be added in the repo. It can be downloaded from: https://drive.google.com/drive/folders/1Yyxh0EVX-5emXX05QFTUOQGv4bFHeFro?usp=sharing
The ‘model’ folder should be put in bgner folder -> bgner/model. 
2.	Make sure that all settings in settings.py and imot_config.py are set
3.	The app is containerized with Docker to make the deployment as easy as possible. To spin the container use:
````
docker-compose build
docker-compose up -d
````

### 1. ETL
The ETL is doing web scraping of the imot.bg web site and is storing the parsed data in Postgres DB. The main steps of
the scrapping are:\
    a)	Collect all the slink URLs (URL that lists all the ads). It has pagination.\
    b)	From the slink get the list of all ads URLs\
    c)	Loop through all ads and scrape the content (includes download of pictures)\
The results are loaded in two tables:\
    a)	ads_latest – contains only the list of active ads with their latest content/ML NERs\
    b)	ads_history – stores all ads (including inactive) and/or previous versions of the ads (can be used as a dataset
        for ML re-training or other ML models)\
**Optimization notes:** Generally the ETL is scheduled to run on a regular basis, however pulling all the ads all the time
    is not very efficient (as we don’t expect to have many new/changed ads) and not very friendly to the web site.
    The following is added to optimize the process:\
    a)	There is a limit of how many ads can be scraped at every run (settings.MAX_ADS_TO_SCRAPE).
        The scrape list would be formed as - first add all net new ads (not scraped before)
        then add all the ads that have not been scraped in the last 24 hours. 
        If the list is longer than the limit, cut it.\
    b)	Once the ads are scraped we would know if the content is already processed 
        (all content is hashed creating unique identifier for the ad). 
        If the hash is already present in ads_history, the ML NER prediction won’t be applied
**Web site structure notes:**
1. Main apartment types: 1 – single bed; 2-double bed; 3-tripple bed; 4-qudruple bed; 5-multiple bed; 6- maisonette;
   7-office; 8-studio/attic; 9-house floor
2. URL construction:
3. f1: page number
4. f7: housing type
5. f38: city
6. f40: housing region
Note: If you want to list multiple values for an f variable use '~'

### 2. NER Model
As the main goal is to be able to more precisely identify the location of the apartment (e.g. street name, 
close to a monument or park…) the model has only 3 labels:\
    a) **B-LOC** (for beginning of location),\
    b) **I-LOC** (for the remainder of the location)\
    c) **O** for non-location tokens. 
All activities around the ML model are encapsulated into the ImotNer class:
1. Predicting for training: This method can leverage any existing model and provide predictions in the same format as
   the needed for the training. Then this file can be manually reviewed/corrected, so a proper labeled set is obtained
2. Training: The actual fine-tuning of the model. It can be used also for continues improvement of an already tuned
   model (especially leveraging the predicting for training method). After every training there is a summary file created
   for the new model with metrics score, data size and update_ts (to keep better track). 
   The parameters of the model are set in imot_config.TRAINING_PARAMS
3. Evaluating existing model (no need to run train first): The method gives comprehensive picture of the performance of
   the model. Accuracy, precision, recall by class and confusion matrix. It also can export the predictions on sub-token
   level for deep-dive review.
4. Predicting for inference: Used for production scoring of ads descriptions

Notes: 
1.	data files are not included. Once generated they should be added in bgner/data folder
2.	The base model used in the fine-tuning is ‘bert-base-multilingual-cased’

### 3. UI
The UI is far from flashy but it provides interactive way to manage the content. It has:\
- Direct link that can open the ad on the side, title picture and some key details of the ad, 
  including the Named Entities. This makes the process of reviewing and disregarding ads very straight forward.\
- Filters by property type\
- ‘Remove’ button that can hide the ad from the UI, but also changes the ‘ad_show’ flag in the DB. 
  This information could be used for training of another model that could give personalized recommendations. Note: If a removed ad had changes in the description or any other ad attribute, it would re-appear.

### 4. DRIVER
It is just a function with the ETL pipeline and the scheduling modul. Set refresh interval can be set here.

### 5. CONFIGURATION
The different parameters should be more or less self-explanatory and commented

Poor man's JIRA: https://miro.com/app/board/uXjVNdbfZd4=/?share_link_id=535992595547

## TODO
    1. Explore serialization of the ML model, so it is not needed to install torch (which takes up to 1 GB)
    2. Improve the UI
    3. Add email notification function for sending latest/most interesting ads.
    





