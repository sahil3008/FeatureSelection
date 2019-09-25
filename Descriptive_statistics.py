import pandas as pd
import numpy as np
import os
import scipy.stats as stas
import matplotlib.pyplot as plt

os.chdir('/Users/sahil308/Desktop/Python data/ICE02/')
os.getcwd()

pollution_data=pd.read_csv('Pollute.txt',sep='\t')
pollution_data.columns

#Finding basic descrptive information for POLLUTION
pollution_data.Pollution.mean()
pollution_data.Pollution.median()
pollution_data.Pollution.min()
pollution_data.Pollution.max()
pollution_data.Pollution.var()
pollution_data.Pollution.std()
print(len(pollution_data.Pollution.unique()))
missing_data= (pollution_data.Pollution.count()-pollution_data.Pollution.size)
print(pollution_data.Pollution.size)

#Finding basic descrptive information for TEMP
pollution_data.Temp.mean()
pollution_data.Temp.median()
pollution_data.Temp.min()
pollution_data.Temp.max()
pollution_data.Temp.var()
pollution_data.Temp.std()
print(len(pollution_data.Temp.unique()))
missing_data= (pollution_data.Temp.count()-pollution_data.Temp.size)
print(pollution_data.Temp.size)

#Finding basic descrptive information for INDUSTRY
pollution_data.Industry.mean()
pollution_data.Industry.median()
pollution_data.Industry.min()
pollution_data.Industry.max()
pollution_data.Industry.var()
pollution_data.Industry.std()
print(len(pollution_data.Industry.unique()))
missing_data= (pollution_data.Industry.count()-pollution_data.Industry.size)
print(pollution_data.Industry.size)

#Finding basic descrptive information for POPULATION
pollution_data.Population.mean()
pollution_data.Population.median()
pollution_data.Population.min()
pollution_data.Population.max()
pollution_data.Population.var()
pollution_data.Population.std()
print(len(pollution_data.Population.unique()))
missing_data= (pollution_data.Population.count()-pollution_data.Population.size)
print(pollution_data.Population.size)

#Finding basic descrptive information for WIND
pollution_data.Wind.mean()
pollution_data.Wind.median()
pollution_data.Wind.min()
pollution_data.Wind.max()
pollution_data.Wind.var()
pollution_data.Wind.std()
print(len(pollution_data.Wind.unique()))
missing_data= (pollution_data.Wind.count()-pollution_data.Wind.size)
print(pollution_data.Wind.size)

#Finding basic descrptive information for RAIN
pollution_data.Rain.mean()
pollution_data.Rain.median()
pollution_data.Rain.min()
pollution_data.Rain.max()
pollution_data.Rain.var()
pollution_data.Rain.std()
print(len(pollution_data.Rain.unique()))
missing_data= (pollution_data.Rain.count()-pollution_data.Rain.size)
print(pollution_data.Rain.size)

#Finding basic descrptive information for Wet.Days
pollution_data['Wet.days'].mean()
pollution_data['Wet.days'].median()
pollution_data['Wet.days'].min()
pollution_data['Wet.days'].max()
pollution_data['Wet.days'].var()
pollution_data['Wet.days'].std()
print(len(pollution_data['Wet.days'].unique()))
missing_data= (pollution_data['Wet.days'].count()-pollution_data['Wet.days'].size)
print(pollution_data['Wet.days'].size)


#Plots for the different variables of pollute dataset
pollution_data.boxplot(column = ['Pollution', 'Temp', 'Rain', 'Wet.days']) 

pollution_data.boxplot(column = 'Wind') 

pollution_data.boxplot(column = 'Industry') 

pollution_data.boxplot(column = 'Population') 

pollution_data.plot.scatter(x = 'Population', y = 'Industry', color = 'green') 

pollution_data.plot.scatter(x = 'Rain', y = 'Wet.days', color = 'blue') 

pollution_data.plot.scatter(x = 'Wind', y = 'Temp', color = 'red') 





pollution_plotting = pd.read_csv('/Users/sahil308/Desktop/Python data/ICE02/ozone.data.txt',sep='\t') 

#QQ Plot For Rad variable:
stas.probplot(pollution_plotting ['rad'], dist = 'norm', plot = plt) 

# Shapiro Wilk Test For Rad variable:
stas.shapiro(pollution_plotting ['rad']) 

#QQ Plot For TEMP variable:
stas.probplot(pollution_plotting ['temp'], dist = 'norm', plot = plt) 

# Shapiro Wilk Test For TEMP variable:
stas.shapiro(pollution_plotting ['temp'])

 #QQ Plot For WIND variable:
stas.probplot(pollution_plotting ['wind'], dist = 'norm', plot = plt) 

# Shapiro Wilk Test For WIND variable:
stas.shapiro(pollution_plotting['wind'])

 #QQ Plot For OZONE variable:
stas.probplot(pollution_plotting ['ozone'], dist = 'norm', plot = plt) 

# Shapiro Wilk Test For OZONE variable:
stas.shapiro(pollution_plotting['ozone']) 


# Histograms for ALL the different variables:

pollution_data ['Pollution'].plot.hist(alpha=0.5) 
pollution_data ['Industry'].plot.hist(alpha=0.5)
pollution_data ['Wind'].plot.hist(alpha=0.5)
pollution_data ['Temp'].plot.hist(alpha=0.5)
pollution_data ['Population'].plot.hist(alpha=0.5)
pollution_data ['Rain'].plot.hist(alpha=0.5)
pollution_data ['Wet.days'].plot.hist(alpha=0.5)
pollution_data.plot.hist()

#Finding out Skewness for the different variables of pollute dataset:

pollution_data['Pollution'].skew()
pollution_data['Temp'].skew()
pollution_data['Industry'].skew()
pollution_data['Wind'].skew()
pollution_data['Population'].skew()
pollution_data['Rain'].skew()
pollution_data['Wet.days'].skew()
pollution_data.skew()

#Finding out Kurtosis for the different variables of pollute dataset:

pollution_data['Pollution'].kurt()
pollution_data['Temp'].kurt()
pollution_data['Industry'].kurt()
pollution_data['Wind'].kurt()
pollution_data['Population'].kurt()
pollution_data['Rain'].kurt()
pollution_data['Wet.days'].kurt()
pollution_data.kurt()





