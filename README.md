# Music Genre Classification

<img src="./images/cover.png" width="1000" alt="photo cover" />

In this project  a dataset with music of different styles such us jazz, classic ... 

## Contents

1. Project Structure
2. Prosess
3. How to run
4. Deployment (Real world Use)
5. To improve
6. About Me

## 1. Project Structure

#### Data
* ├── diabetes.csv
* ├── cleaned_data.csv
* ├── scaled_data.csv
* ├── data_documentation.pdf
#### analysis_and_training
* ├── 1_data_preprocessing
* ├── 2_Music_Genre_Classification_with_ANN.ipynb
* ├── 3_Music_Genre_Classification_with_CNN.ipynb
* ├── 4_Music_Genre_Classification_with_RNN.ipynb
* ├── helpers.py
* ├── models_config.py
#### figures
* ├── contains graph and figures
#### models
* ├── contains trained models
#### images
* ├── contains images used in this images


##### environment
##### requirements.txt
##### gitignore
##### report (pdf & ppt)

## 2. Process

* step1 :  Importing Packages
* step2 :  Loading the data
* step3 : Exploratory Data Analysis (EDA)
* step4 : Data Preparation
* step5 : Build and Train the model
* step6 : Model prediction and Evaluation
* step7 : Model Improvement
 + Hyperparameter Tuning, Features Selection and Features Ingeneering
* step8 : Model Deployment

## 3. How to run

**N.B : python 3.7 is recommended**

### 3.1. CLONE PROJECT DIRECTORY

+ $ git clonehttps://github.com/Rekidiang2/au01_music_genre_classification.git
+ $ cd au01_music_genre_classification

### 3.2. CREATE & ACTIVATE VIRTUAL ENVIRONMENT

#### 3.2.1. WITH PIP and VENV

##### (Windows) 
+ $ python -m venv au01_venv 
+ $ p01ml_venv\Scripts\activate (<= Activate virtual Environment)
+ $ deactivate (<= Deactivate virtual Environment)
+ $ pip install -r requirements.txt
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name au01_venv --display-name "au01_kernel"
+ $ jupyter notebook
+ Go to Kernel -> Change kernel -> au01_kernel

##### (MasOS || LINUX)
+ $ python3 -m venv p01ml_venv 
+ $ source au01_venv/bin/activate (<= Activate virtual Environment)  
+ $ deactivate (<= Deactivate virtual Environment)
+ $ pip install -r requirements.txt
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name au01_venv --display-name "au01_kernel"
+ $ jupyter notebook
+ Go to Kernel -> Change kernel -> au01_kernel


#### 3.2.2. WITH CONDA

+ Verify if you have conda installed ($conda --version) if not go to [anconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) to download and install it

+ $ conda create -n au01_venv python=3.7
+ $ conda activate au01_venv (<= Activate virtual Environment)
+ $ conda deactivate  (<= Deactivate virtual Environment)
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name au01_venv --display-name "au01_kernel"
+ $ jupyter notebook
+ Go to Kernel -> Change kernel -> au01_kernel


#### Manage kernel
+ $ jupyter kernelspec list (<= list all ipykernel in your system)
+ $ jupyter kernelspec uninstall au01_kernel (<= Delete the ipykernel in your system)

## 4. Deployment (Real world Use)

+ [WebApp with Streamlit]()
+ [WebApp with Flask]()


## 5. To improve

+ put link for real world app after deployment in heroku

## 6. About Me
___

### I'm Data Analyst, Data Scientist and Web Developer. Data and technology passionate person, Artificial Intelligence enthusiast. 

> My Website [Click Here](https://rekidiangdata-s.github.io/kiesediangebeni/)

> Social Network

[![alt text][1.1]][1]
[![alt text][2.1]][2]
[![alt text][3.1]][3]
[![alt text][4.1]][4]

[1.1]: https://i.imgur.com/oFsAcMx.png (facebook icon with padding)
[2.1]: https://i.imgur.com/YCdR3o9.png (twitter icon with padding)
[3.1]: https://i.imgur.com/5BWvIrF.png (github icon with padding)
[4.1]: https://i.imgur.com/UA7Oh6z.png (medium icon with padding)

[1]: http://www.facebook.com/reagan.kiese.37
[2]: https://twitter.com/ReaganKiese
[3]: https://github.com/RekidiangData-S
[4]: https://medium.com/@rkddatas

