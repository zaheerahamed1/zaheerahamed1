#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark



from pyspark import SparkContext, SparkConf


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
spark = (SparkSession.builder.appName("project").config("hive.metastore.uris","thrift://ip-10-1-2-24.ap-south-1.compute.internal:9083").enableHiveSupport().getOrCreate())


# In[3]:


#spark.stop()


# In[4]:


spark = SparkSession.builder.appName('sunny')        .config("hive.metastore.uris","thrift://ip-10-1-2-24.ap-south-1.compute.internal:9083")        .enableHiveSupport().getOrCreate()
spark


# In[5]:


spark.sql("use samreenalab_capstone").show()


# In[6]:


employees = spark.sql('select * from samreenalab_capstone.employees')

employees.show()


# In[7]:


departments = spark.sql('select * from samreenalab_capstone.departments')

departments.show()


# In[8]:


department_employees = spark.sql("select * from samreenalab_capstone.department_employees")

department_employees.show()


# In[9]:


department_managers = spark.sql('select * from samreenalab_capstone.department_employees')

department_managers.show()


# In[10]:


titles = spark.sql('select * from samreenalab_capstone.titles')

titles.show()


# In[11]:


salaries = spark.sql('select * from samreenalab_capstone.salaries')

salaries.show()


# In[ ]:





# In[ ]:





# In[12]:


# salaries = spark.read.csv('capstone/salaries.csv', header = True, inferSchema = True)

# salaries.show()

# departments = spark.read.csv('capstone/departments.csv', header = True, inferSchema = True)

# departments.show()


# titles = spark.read.csv('capstone/titles.csv', header = True, inferSchema = True)

# titles.show()

# department_managers = spark.read.csv('capstone/dept_manager.csv', header = True, inferSchema = True)

# department_managers.show()     

# department_employees = spark.read.csv('capstone/dept_emp.csv', header = True, inferSchema = True)

# department_employees.show()

# employees = spark.read.csv('capstone/employees.csv', header = True, inferSchema = True)

# employees.show()


# In[13]:


df = employees.join(department_employees, on = 'emp_no', how = 'left').join(departments, on = 'dept_no', how = 'left').join(salaries, on = 'emp_no', how ='left').join(titles, employees.emp_title_id == titles.title_id, 'left')


df.show()


# # Exploratory Data Analysis

# In[14]:


# dimension of the data frame


print("th DIM of DF : ",(df.count(), len(df.columns)))


# In[15]:


# columns in DF

df.columns


# In[16]:



df.describe().show()


# In[17]:


df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

#Data Analysis with Python(Pandas)
# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (14,8)


# In[19]:


pd_df = df.toPandas()


# In[20]:


pd_df.head()


# In[21]:


pd_df.isnull().sum()


# In[22]:


# number of  employees

pd_df['emp_no'].nunique()


# In[23]:


# number of departments

print('number of dept : ', pd_df['dept_no'].nunique())

pd_df['dept_name'].unique()


# In[24]:


df.printSchema()


# # Data Visualization

# In[25]:


sns.countplot(pd_df['dept_name'])
plt.xticks(rotation = 90)
plt.show(1)


# In[26]:


#sns.countplot(pd_df['left_'], hue = pd_df['dept_name'], color = pd_df['sex'])


# In[27]:


pd_df.groupby('dept_name')[['salary']].max()


# In[28]:


pd_df.groupby('dept_name')[['salary']].min()


# In[29]:


pd_df.groupby('dept_name')[['salary']].mean()


# In[30]:


sns.distplot(pd_df['salary'])
#plt.ylabel()
plt.title('salary ')
plt.show()

#Salary Has Skewed data we need to apply some transformations on salary columns like log(), exp(), square(),....
# In[31]:


pd_df.columns


# In[32]:


sns.countplot(pd_df['left_'])
plt.show()

#The Given Data Set is Imabalanced Data Set, 
#We can Not Build Model This Type of data 
#Set So We Need to convert it into Balanced data Set
# In[33]:


sns.countplot(pd_df['left_'], hue = pd_df['sex'])
plt.title('Churn The Employees by gender Wise')
plt.show()


# In[34]:


sns.countplot(pd_df['left_'], hue = pd_df['dept_name'])
plt.title("Churn the Employees by Department Wise")
plt.show()

#From the Department Production and Sales Has More Chances To Churn(left) The Employees
# In[35]:


pd_df.groupby('emp_no')[['no_of_projects']].sum().sort_values(by = 'no_of_projects',ascending = False)


# In[36]:


pd_df.groupby('dept_name')[['no_of_projects']].sum()


# In[37]:


pd_df.groupby('dept_name')[['no_of_projects']].sum().plot(kind = 'bar', color = 'pink', edgecolor = 'black')

#Departments of Development and Production has done Maximum Projects
# In[38]:


sns.countplot(pd_df['dept_name'], hue = pd_df['sex'])

#In Every Department Males Are Maximum Than Females
# In[39]:


pd_df['Last_performance_rating'].unique()


# In[40]:


sns.countplot(pd_df['Last_performance_rating'])


# In[41]:


sns.countplot(pd_df['Last_performance_rating'], hue = pd_df['sex'])


# In[42]:


title_salary = pd_df.groupby('title')['salary'].sum()

plt.pie(title_salary,autopct="%1.0f%%", colors = ['pink','blue','green'])
plt.title('salary per title')

plt.show()


# In[43]:


pd_df.groupby(['dept_name','Last_performance_rating'])[['emp_no']].count()


# In[44]:


df.show()


# In[45]:


true_df = pd_df[pd_df['left_'] == 1]

false_df = pd_df[pd_df['left_'] == 0]


# In[46]:


true_df.head()


# In[47]:


false_df.head()


# In[48]:


print('DIM of True DF : ', (true_df.shape))

print('DIM of False DF : ',(false_df.shape))


# In[49]:


false_samp_df = false_df.sample(true_df.shape[0])

false_samp_df


# In[50]:


final_df = pd.concat([true_df,false_samp_df])

final_df


# In[51]:


from sklearn.utils import shuffle


# In[52]:


final_df = shuffle(final_df)


# In[53]:


final_df


# In[54]:


final_df['left_'].value_counts()


# In[55]:


sns.countplot(final_df['left_'])

#Now its Became Balanced data Set Now We can Create Machine Learning Model On the Final DF
# # Machine Learning Model for Predict The Employee left/not

# In[56]:


from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[57]:


final_df.describe().T


# In[58]:


ml_data = spark.createDataFrame(final_df)

ml_data.show()


# In[59]:


type(ml_data)


# In[60]:


ml_data.columns


# In[61]:


ml_data.printSchema()


# In[62]:


pd_df.head()

pd_df['age'] = [i[2] for i in pd_df['birth_date'].str.split("/")]

pd_df['age'] = pd_df['age'].astype('int')


# In[63]:


pd_df['left_'] = [str(i).replace('True','1').replace('False','0') for i in pd_df['left_']]

pd_df['left_'] = pd_df['left_'].astype('int')

pd_df['left_'].dtype


# In[64]:


pd_df.head()


# In[65]:


ml_data = spark.createDataFrame(pd_df)

ml_data.show()


# In[66]:


type(ml_data)

#ML Features : dept_no, sex, Last_performance rating, title these are categorical variables


#ML Features : no_of_projects, salary, age, these are numerical variables
    

#ML target var : left_

# In[ ]:





# In[67]:


# create object of StringIndexer class and specify input and output column
SI_dept_name = StringIndexer(inputCol='dept_name',outputCol='dept_name_Indexed')
SI_dept_no = StringIndexer(inputCol='dept_no',outputCol='dept_no_Indexed')
SI_title = StringIndexer(inputCol='title',outputCol='title_Indexed')
SI_lpr = StringIndexer(inputCol = 'Last_performance_rating', outputCol = 'lpr_Indexed')



# transform the data
ml_data = SI_dept_name.fit(ml_data).transform(ml_data)
ml_data = SI_dept_no.fit(ml_data).transform(ml_data)
ml_data = SI_title.fit(ml_data).transform(ml_data)
ml_data = SI_lpr.fit(ml_data).transform(ml_data)



# view the transformed data
ml_data.select('dept_name', 'dept_name_Indexed', 'dept_no', 'dept_no_Indexed', 'title','title_Indexed','Last_performance_rating','lpr_Indexed').show(5)


# In[68]:



from pyspark.ml.feature import OneHotEncoderEstimator

# create object and specify input and output column
OHE = OneHotEncoderEstimator(inputCols=['dept_name_Indexed', 'dept_no_Indexed','title_Indexed','lpr_Indexed'],outputCols=['dept_name_vect', 'dept_no_vect','title_vect','lpr_vect'])

# transform the data
ml_data = OHE.fit(ml_data).transform(ml_data)

# view and transform the data
ml_data.select('dept_name', 'dept_name_Indexed', 'dept_name_vect', 'dept_no', 'dept_no_Indexed', 'dept_no_vect', 'title','title_Indexed','title_vect').show(10)


# In[70]:


#Columns that will be used as features and their types
continuous_features = ['salary','no_of_projects','age']
                    
categorical_features = ['dept_name', 'dept_no','title','Last_performance_rating']


# In[71]:


featureCols = continuous_features + ['dept_name_vect', 'dept_no_vect', 'title_vect','lpr_vect'] 


# In[72]:



assembler = VectorAssembler(inputCols = featureCols, outputCol = "features")


# In[73]:


ml_data.show()


# In[74]:


ml_data = assembler.transform(ml_data)


# In[75]:


ml_data.select('features').show(5)


# In[76]:


ml_data.printSchema()


# In[77]:


#Split the dataset
train_data, test_data = ml_data.randomSplit( [0.8, 0.2], seed = 194 )


# In[78]:


print("DIM OF Train DF : ", (train_data.count(), len(train_data.columns)))

print("DIM OF Train DF : ", (test_data.count(), len(test_data.columns)))


# In[79]:


print("DF type : ",type(train_data))
print("DF type : ",type(test_data))


# In[80]:


logit_model = LogisticRegression(featuresCol = 'features', labelCol = 'left_', maxIter = 10)


# In[81]:


sigmoid_model = logit_model.fit(train_data)


# In[82]:


pred_train_data = sigmoid_model.transform(train_data)

pred_test_data = sigmoid_model.transform(test_data)


# In[83]:


pred_train_data.select('features','left_','prediction').show()


# In[84]:


pred_test_data.select('features','left_','prediction').show()


# In[85]:


train_accuracy = pred_train_data.filter(pred_train_data.left_ == pred_train_data.prediction).count() / float(pred_train_data.count())
print("Tain Data Accuracy : ",train_accuracy)


# In[86]:


test_accuracy = pred_test_data.filter(pred_test_data.left_ == pred_test_data.prediction).count() / float(pred_test_data.count())
print("Test Data Accuracy : ",test_accuracy)


# In[87]:


# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# evaluator = BinaryClassificationEvaluator()


# evaluator.evaluate(pred_train_data)


# In[88]:


ml_data.show()


# In[89]:


rfc_ml_data = ml_data

rfc_ml_data.show()


# In[90]:


train_data,test_data = rfc_ml_data.randomSplit([0.8,0.2], seed = 135)


# In[91]:


RFC = RandomForestClassifier(featuresCol = 'features', labelCol = 'left_')

RFC_model = RFC.fit(train_data)


# In[92]:


pred_train_data = RFC_model.transform(train_data)

pred_train_data = RFC_model.transform(test_data)


# In[93]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="left_", predictionCol="prediction", metricName="accuracy")
train_accuracy = evaluator.evaluate(pred_train_data)
print("Train Error = %g" % (1.0 - train_accuracy))


# In[94]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="left_", predictionCol="prediction", metricName="accuracy")
test_accuracy = evaluator.evaluate(pred_test_data)
print("Test Error = %g" % (1.0 - test_accuracy))


# # PipeLine Making For Binary Classification

# In[ ]:


pipeline_data = spark.createDataFrame(pd_df)

pipeline_data.show()


# In[ ]:


type(pipeline_data)


# In[ ]:


pipeline_data.printSchema()


# In[ ]:


labelIndexer = StringIndexer(inputCols = ['dept_name','titlte','dept_no'], outputCol = ['dept_name_index','title_index','dept_no_index']).fit(pipeline_data)
labelIndexer.transform(pipeline_data).show(5, True)


# In[ ]:


# one_HE = OneHotEncoderEstimator(inputCols = ['dept_name_index','title_index','dept_no_index'], outputCols = ['dept_name_vect','title_vect','dept_no_vect']).fit(pipeline_data)
# one_HE.transform(pipeline_data).show(5,True)


# In[ ]:


featureIndexer =VectorIndexer(inputCol = "features",                                   outputCol="indexedFeatures",                                   maxCategories=4).fit(data)
featureIndexer.transform(pipeline_data).show(5, True)


# In[ ]:


(trainingData, testData) = data.randomSplit([0.6, 0.4])

trainingData.show(5,False)

testData.show(5,False)


# In[ ]:


logr = LogisticRegression(featuresCol='features', labelCol='left_')


# In[ ]:


# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",labels=labelIndexer.labels)


# In[ ]:


# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr,labelConverter])


# In[ ]:


# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)


# In[ ]:


# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("features","label","predictedLabel").show(5)


# In[203]:


from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression


categorical_features = ['dept_name','dept_no','title']
continious_features = ['age','salary','no_of_projects']

## Create indexers for the categorical features
indexers = [StringIndexer(inputCol = c, outputCol="{}_idx".format(c)) for c in categorical_features]

## encode the categorical features
encoders = [OneHotEncoderEstimator(inputCol = idx, outputCol = "{0}_enc".format(idx)) for idx in indexers]

## Create vectors for all features categorical and continuous

assembler = VectorAssembler(inputCols=[enc.getOutputCol() for enc in encoders] + continuous_features, outputCol = "features")

## Initialize the linear model
lrModel = LogisticRegression( maxIter = 10 )


## Create the pipeline with sequence of activities
#pipeline = Pipeline( stages=indexers + encoders + [assembler, lrModel ])

pipeline = Pipeline( stages= [indexers, Estimator, assembler, lrModel ])


# In[ ]:


chrun_pipeline_df = pipeline_data.withColumn( 'label', 'left_' )


# In[ ]:


training, testing = pipeline_data.randomSplit( [0.7, 0.3], seed = 42 )


# In[ ]:





# In[ ]:




