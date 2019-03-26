Statistical learning refers to a vast set of tools for understanding data. These tools can be classified as supervised or unsupervised. 
The following supervised statistical learning models for predicting, or estimating, an output based on one or more inputs, illustrate various concepts of machine learning as I understand them. 


* **Project 1: Data Collection - Web Scraping - Data Parsing - IMDB Top Stars Analysis**

In the first part of the project, we parse the HTML page of a professor containing some of his/her publications, and convert to bibTex.  answer some questions.
We use Python BeautifulSoup library, python dictionary, dataframe, OS files.

In the second part of the project, we extract information from [IMDb's Top 100 Stars for 2017](https://www.imdb.com/list/ls025814950/) and perform some analysis. We use Python list, dictionary, JSON file, matplotlib, regular expression, next to BeautifulSoup.

* **Project 2: Linear and k-NN Regression - Predicting Taxi Pickups in NYC**

We build regression models that can predict the number of taxi pickups in New York city at any given time of the day. We use matplotlib and pandas to visualize the data, KNeighborsRegressor from sklearn for regression. We also built OLS model from statsmodel and study parameters with confidence levels. We investigate how outliers affect the linear regression model and develop a outliers removal algorithm.

* **Project 3: Exploratory Data Analysis - Forecasting Bike Sharing Usage**

In this project we predict the hourly demand for rental bikes and give suggestions on how to increase their revenue. We explore the data using pandas scatter matrix, seaborn violin plots. We engineer additional time features and identify outliers using seaborn box plots.
We one-hot-encode qualitative features, create design matrixes and perform mult-linear regression using statsmodel OLS, R2 as metric. We use p-values to classify less significant predictors. We study the residuals using histograms and scatter plots.
We also investigate polynomial and interaction terms and study the effect of multicolinearity. We implement forward-step-wise features selection and select a final model for our recommendations.

* **Project 4: Regularization - Ridge - Lasso - Forecasting Bike Sharing Usage**

In this project, we extend project 2's models by applying regularization techniques. We compare Ridge with Lasso regressions with different lambdas. We study the coefficients histograms and the effect of regularization on their magnitudes.

* **Project 5: Logistic Regression, High Dimensionality and PCA - Cancer Classification from Gene Expressions**

In this project, we will build a classification model to distinguish between two related classes of cancer, acute lymphoblastic leukemia (ALL) and acute myeloid leukemia (AML), using gene expression measurements.
We explore how each gene differentiates the classes using histograms. We use PCA to find the number of components which explain data variability. 
We fit a simple linear regression model using one gene and derivate the class from the regression line. We compare this naive model with a logistic model.
We also fit a multiple logistic regression using all genes and visualize the probabilities obtained. We compare the model with a logistic regression fit on PCA components. 

* **Project 6: Multilayer Feedforward Network - Dealing with Missing Data in Pima Indians onset of diabetes**

In the first part of the project we construct feed forward neural networks by hand: perceptron layer with affine transformation, linear output layer, manual weights searching, sigmoid log loss, manual gradient descent implementation.
We use the network to approximate a step, a one hump and a two humps funtion.

In the second part, we deal with missing data in the patient medical record data for Pima Indians and whether they had an onset of diabetes within five years. We detect 
values used for missing data, and replace those values with NaN. We implent different methods for missing values imputation: by dropping the column, by dropping the row, by applying kNN neighbors while using features without NaN as predictors.
We compare the classification accuracy of a logistic regressor after applying those different imputation methods and cross-validation.

* **Project 7: Classification with Logistic Regression, LDA/QDA, and Trees - Multiclass Thyroid Classification**

In the project we build a model for diagnosing disorders in a patient's thyroid gland (normal, hyperthyroidism, hypothyroidism). We deal with stratified train-test split. We create and benchmark the following models:

- a baseline model that predicts the majority class and plot the decision boundaries.
- a logistic regression with L2 regularization tuned via cross-validation. 
- a logistic regression with quadratic terms.
- a linear discriminant analysis model. We evaluate the similarity between LDA and logistic regression.
- a quadratic discriminant analysis model. We visualize the effect of covariance among classes on the quadratic decision surfaces.
- a decision tree with depth choosen via cross-validation. We visualize the decision tree, especially how to navigate from root to leaves when diagosing diabetes.
- a kNN model with number of neigbors obtained via cross-validation. 

Additionally to model selection, we study how to mitigate miss-diagnosis by allowing the model to 'abstain' from making a prediction: whenever it is uncertain about the diagnosis for a patient. 
However, when the model abstains from making a prediction, the hospital will have to forward the patient to a thyroid specialist (i.e. an endocrinologist), which would incur additional cost.  We suggest a design of thyroid classification model with an abstain option, such that the cost to the hospital is minimized.

* **Project 8: Ensembles: Bagging, Random Forests, and Boosting - Higgs boson particle Classification**


In this project we use ensemble methods to differentiate between collisions that produce Higgs bosons and collisions that produce only background noise.

We find the optimal depth of a single decision tree that produces the best classification accuracy via cross-validation. We create several bootstraps of the data and manually build a bagging model of the tree. We exhibit the bagging trees correlation issue and introduce random forests. We display how often predictors appear at the biginning of the trees, showing the supperior de-correlation effect by random forests.
Next we create a manual boosting model and compare it to an AdaBoost model.

* **Project 9: ANNs, Keras, Regulrizing Neural Networks, MNIST handwritten digits**

In this project we use Keras to built a fully connected multilayered perceptron that approximates any function to a set of sine and cosine base functions as done by Frourier Transformation.

We also build a FCNN to classify hand-written digits from the MNIST dataset. We then explore several hyperparameters tuning options and regularization procedures: optimizer, number of epochs, batch size, learning rate; data augmentation by image transformation, early stopping, dropout, L2, L1.