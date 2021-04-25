# Know Your Data

[![Build Status](https://camo.githubusercontent.com/f9010d0d18143896d2e496fe0e0c89056acab8229dbdf169f1d3a4759567fe63/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d616465253230776974682d507974686f6e2d3166343235662e737667)](https://github.com/SohamChhapre/Final-Project-Server)
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/SohamChhapre/Final-Project-Server)

Know Your Data is Multi Variate Data Analysis Platform.



## Features

- Text Summary , Topic Modelling,Polarity Score ,Fog Index, Uncertainity Score, constraning proportion,uncertainity proportion,negative proportion,positive proportion of document
- Summary of Text, Topic Modelling,Polarity Score ,Fog Index, Uncertainity Score, constraning proportion,uncertainity proportion,negative proportion,positive proportion of scanned document.
- Error Sampling rate in video (to detect anamoly)
- Speaker Diarization of audio , Transcripts of diffrent speakers,intervals of speakers 
- Generate report of text analysis









## Installation

Know Your Data requires [Python](https://www.python.org/downloads/release/python-370/) v 3.7(recommended) to run.

Install the dependencies and devDependencies and start the server.

#### Tessarect OCR installation:

In order to use the Tesseract library, we first need to install it on our system.

To download tesseract ocr for python open the below link in your browser

```sh
https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20200328.exe
```

To validate that Tesseract has been successfully installed on your machine, execute the following command:

```sh
tesseract -v
```

You should see the Tesseract version printed to your screen, along with a list of image file format libraries Tesseract is compatible with.
If output is diffrent please ensure that the path of tesseract.exe is set on environment variable, If not set the path in environment variable. 

#### Clonning a Github Repo:
```sh
git clone https://github.com/SohamChhapre/Final-Project-Server.git
```

#### Installing Virtual Enviromnment:

```sh
pip install virtualenv
```
#### Create Virtual Environment:
```sh
virtualenv .env
```
#### Activating Virtual Environment:
```sh
cd path_to_.env/Scripts
activate
```
#### Installing dependencies:
```sh
pip install -r requirements.txt
```

#### Running Server:
```sh
python Server.py
```
Server is Deployed at:
```sh
127.0.0.1:5000
```

## Contributors:
- [Anshul Bhatia](https://github.com/anshul-bhatia)
- [Chaitanya Shrivastava](https://github.com/artist1327)
- [Shivam Kumar Mahto](https://github.com/shivam0403)
- [Soham Chhapre](https://github.com/SohamChhapre)
- [Vishal Rochlani](https://github.com/wish15)
