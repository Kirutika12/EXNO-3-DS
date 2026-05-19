## EXNO-3-DS

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
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="511" height="378" alt="image" src="https://github.com/user-attachments/assets/18197d0f-2e00-4cb5-be87-7132be04361c" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

<img width="207" height="216" alt="image" src="https://github.com/user-attachments/assets/3eaae68e-2761-4a39-9d50-9db2facb513c" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```


<img width="467" height="360" alt="image" src="https://github.com/user-attachments/assets/f5556529-c584-4dce-9a26-d0f6d763447e" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="383" height="367" alt="image" src="https://github.com/user-attachments/assets/7828f989-8923-4c58-b7f1-70a27638656a" />

```
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)   
df2 = df.copy()

enc = pd.DataFrame(
    ohe.fit_transform(df2[["nom_0"]]),
    columns=ohe.get_feature_names_out(["nom_0"]),
    index=df2.index
)

df2 = pd.concat([df2, enc], axis=1)
df2
```

<img width="701" height="367" alt="image" src="https://github.com/user-attachments/assets/25003267-d42e-44eb-bc09-c246a16dac50" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="958" height="372" alt="image" src="https://github.com/user-attachments/assets/dc6eb305-ae6c-42ce-9857-cef033931d60" />

```
pip install --upgrade category_encoders
```

<img width="1247" height="588" alt="image" src="https://github.com/user-attachments/assets/07820a4b-a78e-4896-9b3c-46aea223057a" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

<img width="512" height="368" alt="image" src="https://github.com/user-attachments/assets/66b5bbf4-bb55-4c53-a648-a13dc7a138c1" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="528" height="368" alt="image" src="https://github.com/user-attachments/assets/3cb20969-186e-46e2-9214-5f7c27fb7eed" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="742" height="365" alt="image" src="https://github.com/user-attachments/assets/ae90d5cf-f196-4b52-8068-2e92fe32b2fd" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="560" height="363" alt="image" src="https://github.com/user-attachments/assets/4a3669b2-e1ad-4e81-8056-0fed568ded6d" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="828" height="447" alt="image" src="https://github.com/user-attachments/assets/f1811a1e-078e-4d3f-856b-11bf8fb9e537" />


```
df.skew()
```

<img width="352" height="115" alt="image" src="https://github.com/user-attachments/assets/552d55e9-a4a6-46c4-9d34-ccc36d39e1cf" />


```
np.log(df["Highly Positive Skew"])
```

<img width="606" height="262" alt="image" src="https://github.com/user-attachments/assets/726f129b-29f3-43bb-978f-58ce4483b753" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="657" height="257" alt="image" src="https://github.com/user-attachments/assets/5d0b2db2-3188-40d0-9e2d-d8c70ae6e9a6" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="656" height="265" alt="image" src="https://github.com/user-attachments/assets/55fe239f-ed37-48e6-a381-e893e82443ec" />

```
np.square(df["Highly Positive Skew"])
```

<img width="617" height="261" alt="image" src="https://github.com/user-attachments/assets/fe17fdba-42a7-4d7b-a402-f46da06ca51c" />

```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```

<img width="755" height="456" alt="image" src="https://github.com/user-attachments/assets/ebaaf865-44f2-4d22-af99-e324114082e8" />

```
 df.skew()
```

<img width="462" height="143" alt="image" src="https://github.com/user-attachments/assets/6fbd9271-e31e-45e1-b550-510d73c2201e" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="440" height="155" alt="image" src="https://github.com/user-attachments/assets/3d5c655f-9e22-4d03-8cf6-a08348fa3f47" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="753" height="472" alt="image" src="https://github.com/user-attachments/assets/3ba3d67b-5956-432e-b830-f1eec898ce8b" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="768" height="555" alt="image" src="https://github.com/user-attachments/assets/cf29c551-be25-407d-a05e-80e438798305" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="765" height="543" alt="image" src="https://github.com/user-attachments/assets/7340c3f7-4fd5-4274-8566-6979384efcdb" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="753" height="547" alt="image" src="https://github.com/user-attachments/assets/6203316b-11ab-46ad-9dbf-0adffa4e02d1" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="747" height="543" alt="image" src="https://github.com/user-attachments/assets/a60ebc6a-e4b5-487f-82d0-d5d2a206fb85" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```

<img width="1243" height="437" alt="image" src="https://github.com/user-attachments/assets/d70e4647-c269-4f49-8e3c-da46239d1dca" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="728" height="545" alt="image" src="https://github.com/user-attachments/assets/72c53d78-aab9-4164-9912-3d9760a0389c" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="737" height="537" alt="image" src="https://github.com/user-attachments/assets/29cc34dc-377b-4493-8169-3fcd0fd98d36" />

# RESULT:
       Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.

       
