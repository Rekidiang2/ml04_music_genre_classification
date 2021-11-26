# Diabetes Prediction

<img src="./images/cover.png" width="1000" alt="photo cover" />


In healthcare field diagnose a problem early offer more chance for traitement and guerison in this project we apply machine learning techniques to predict whether a patient will develop diabetes within the next five years. Early detection and diagnosis of diabetes is that the early stages of diabetes are often non-symptomatic. People who are on the path to diabetes (also known as prediabetes) often do not know that they have diabetes until it is too late.

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
* ├── py_files
* ├── 1_data_preprocessing
* ├── 2_ML_model
* ├── 3_DL_model
* ├── helpers.py
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

+ $ git clonehttps://github.com/RekidiangData-S/p01ml_diabetes_prediction.git
+ $ cd p01ml_diabetes_prediction

### 3.2. CREATE & ACTIVATE VIRTUAL ENVIRONMENT

#### 3.2.1. WITH PIP and VENV

##### (Windows) 
+ $ python -m venv p01ml_venv 
+ $ p01ml_venv\Scripts\activate (<= Activate virtual Environment)
+ $ deactivate (<= Deactivate virtual Environment)
+ $ pip install -r requirements.txt
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name p01ml_venv --display-name "p01ml_kernel"
+ $ jupyter notebook

##### (MasOS || LINUX)
+ $ python3 -m venv p01ml_venv 
+ $ source p01ml_venv/bin/activate (<= Activate virtual Environment)  
+ $ deactivate (<= Deactivate virtual Environment)
+ $ pip install -r requirements.txt
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name p01ml_venv --display-name "p01ml_kernel"
+ $ jupyter notebook


#### 3.2.2. WITH CONDA

+ Verify if you have conda installed ($conda --version) if not go to [anconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) to download and install it

+ $ conda create -n p01ml_venv python=3.7
+ $ conda activate p01ml_venv (<= Activate virtual Environment)
+ $ conda deactivate  (<= Deactivate virtual Environment)
+ Set  VIRTUAL ENVIRONMENT as KERNEL : 
  +  $ python -m ipykernel install --user --name p01ml_venv --display-name "p01ml_kernel"
+ $ jupyter notebook
+ Go to Kernel -> Change kernel -> p01ml_kernel
+ $ jupyter kernelspec list (<= list all ipykernel in your system)
+ $ jupyter kernelspec uninstall p01ml_venv (<= Delete the ipykernel in your system)


#### Manage kernel
+ $ jupyter kernelspec list (<= list all ipykernel in your system)
+ $ jupyter kernelspec uninstall p01ml_venv (<= Delete the ipykernel in your system)

## 4. Deployment (Real world Use)

+ [WebApp with Streamlit]()
+ [WebApp with Flask]()


## 5. To improve

+ put link for real world app after deployment in heroku

## 6. About Me
___

### I'm a data scientist, software Engineer. data and technology passionate person, Artificial Intelligence enthusiast 

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

