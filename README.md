## EXNO-3-DS
## Shreenidhi S 212225040410
 
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~

<img width="1917" height="1013" alt="Screenshot 2026-08-06 201159" src="https://github.com/user-attachments/assets/a2bb2081-6e60-4d17-8761-2fc3a8977282" />

~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~

<img width="1917" height="865" alt="Screenshot 2026-08-06 201212" src="https://github.com/user-attachments/assets/0a0c14e9-30d3-4793-a8bb-3eba354a5b4d" />

~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~

<img width="1917" height="1017" alt="Screenshot 2026-08-06 201226" src="https://github.com/user-attachments/assets/4f30b01c-fdde-42a2-b6fd-77ccb7a289b9" />

~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc["ord_2"])
dfc
~~~

<img width="1916" height="1007" alt="Screenshot 2026-08-06 201238" src="https://github.com/user-attachments/assets/4e9164ad-2db1-497f-95d9-f44182091c8c" />

~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2=pd.concat([df2,enc],axis=1)
df2
~~~

<img width="1912" height="1010" alt="Screenshot 2026-08-06 201254" src="https://github.com/user-attachments/assets/1b5ee396-4df8-48fc-9e40-a267700809c4" />

~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~

<img width="1917" height="1018" alt="Screenshot 2026-08-06 201309" src="https://github.com/user-attachments/assets/88b62cf7-cd0e-45a9-be60-2bae0306a8d5" />
~~~
pip install --upgrade category_encoders
~~~

<img width="1917" height="1012" alt="Screenshot 2026-08-06 201321" src="https://github.com/user-attachments/assets/51cf09c7-4593-46bd-aa4e-ebe929db6728" />

~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb=pd.concat([df,nd],axis=1)
dfb
~~~

<img width="1917" height="1017" alt="Screenshot 2026-08-06 201331" src="https://github.com/user-attachments/assets/0abf3e21-3c11-4e0e-8d51-3bf13346bf28" />

~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
~~~

<img width="1917" height="997" alt="Screenshot 2026-08-06 201346" src="https://github.com/user-attachments/assets/c9f4e083-4393-4ee1-b779-c48262edb8b1" />

~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
~~~

<img width="1917" height="1013" alt="Screenshot 2026-08-06 201358" src="https://github.com/user-attachments/assets/16fa4a21-7872-401b-bfb7-14f9c4153d76" />

~~~
np.log(df["Highly Positive Skew"])
~~~

<img width="1917" height="1008" alt="Screenshot 2026-08-06 201414" src="https://github.com/user-attachments/assets/0bd6a104-e341-4ed4-b3ed-6352f17a61e6" />

~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~

<img width="1916" height="1011" alt="Screenshot 2026-08-06 201424" src="https://github.com/user-attachments/assets/514dc35c-cab2-4a8c-9e1b-b29a23e56db3" />
~~~
np.sqrt(df["Highly Positive Skew"])
~~~

<img width="1912" height="1012" alt="Screenshot 2026-08-06 201458" src="https://github.com/user-attachments/assets/9637d9c7-7c9f-4b0e-b027-44703281ab6c" />
~~~
np.square(df["Highly Positive Skew"])
~~~

<img width="1917" height="1012" alt="Screenshot 2026-08-06 201524" src="https://github.com/user-attachments/assets/ba5f01cd-c87f-4002-801b-930e9f54bd41" />

~~~
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
~~~

<img width="1912" height="995" alt="Screenshot 2026-08-06 201534" src="https://github.com/user-attachments/assets/6dfaac11-c5bd-481f-8eca-f0c02e6a90ed" />
~~~
df.skew()
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])

df.skew()
~~~

<img width="1912" height="1011" alt="Screenshot 2026-08-06 201557" src="https://github.com/user-attachments/assets/a0d25339-a780-47ec-9392-bd20f398891f" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
 
<img width="1916" height="1021" alt="Screenshot 2026-08-06 201545" src="https://github.com/user-attachments/assets/1b2055c1-112c-4e78-b256-953635eb3e11" />

~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()
~~~

<img width="1915" height="1008" alt="Screenshot 2026-08-06 201900" src="https://github.com/user-attachments/assets/af24532d-a89d-4db7-a929-4b1fb90ced33" />

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]), line='45')
plt.show()
~~~

<img width="1917" height="1015" alt="Screenshot 2026-08-06 201910" src="https://github.com/user-attachments/assets/9b9a3681-e056-439f-a295-d0a5fda53eab" />

~~~
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)

df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()
~~~


<img width="1916" height="1013" alt="Screenshot 2026-08-06 201938" src="https://github.com/user-attachments/assets/93540a13-8f09-4df0-b5b3-2320abfc1bd5" />

~~~
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"], line='45')
plt.show()
~~~

<img width="1917" height="1012" alt="Screenshot 2026-08-06 201949" src="https://github.com/user-attachments/assets/832f685e-0d4b-4498-a5be-ae7600e27573" />

~~~
sm.qqplot(df["Highly Negative Skew_1"], line='45')
plt.show()
~~~

<img width="1916" height="1017" alt="Screenshot 2026-08-06 202001" src="https://github.com/user-attachments/assets/e49fea11-cb33-4138-b0da-15144ed5fe71" />

~~~
dt=pd.read_csv("/content/titanic_dataset.csv")
dt["Age_1"] = qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"], line='45')
plt.show()
~~~

<img width="1916" height="1015" alt="Screenshot 2026-08-06 202012" src="https://github.com/user-attachments/assets/735fd4cc-d30f-4a72-8eb8-2f29e20a5693" />

~~~
sm.qqplot(dt["Age_1"], line='45')
plt.show()
~~~

<img width="1912" height="1001" alt="Screenshot 2026-08-06 202026" src="https://github.com/user-attachments/assets/c04d48d1-fc87-456a-a5ec-9a12ce22a6cb" />


# RESULT:
   Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.
       
