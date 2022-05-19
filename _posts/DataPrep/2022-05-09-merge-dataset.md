---
title:  "Merge various datasets with Pandas"
categories:
  - Data-tab
tags:
  - Pandas
  - Tabular data
  - Data manipulation
  - Merge
  
classes: wide
toc: false

---

Here, I'm presenting my code I used to manipulate and merge multiple datasets.
This is part of my [dancer's business](https://minjung-mj-kim.github.io/projects-db/dancer/) project.
Hope it is useful!

-----------------
------------------

# Load libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# DataFrame to save


```python
# Dataframe to merge everything
df = None
```

# Scale/adjustment factor data
Before we begin, let's import a few useful economic index data and census data.
Even though these data won't be used for my modeling, 
I will use them when we explore statistics to get a trend of business.

## Consumer Price Index (CPI)
CPI can be used to account for inflation. I downloaded numbers of interesting areas from the website below.
- Source: [U.S. Bureau of Labor Statistics (link is one of the example page)](https://www.bls.gov/regions/new-york-new-jersey/data/xg-tables/ro2xgcpiny1967.htm)
    - All area: Mean of half1 and half2 of "All items in U.S. city average, all urban consumers, not seasonally adjusted" (CUUR0000SA0)



```python
cpi = pd.read_csv('data/CPI.csv')
display(cpi)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>NY</th>
      <th>Chicago</th>
      <th>Seattle</th>
      <th>LA</th>
      <th>SanFran</th>
      <th>All</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1990</td>
      <td>138.500</td>
      <td>131.700</td>
      <td>126.800</td>
      <td>135.900</td>
      <td>132.100</td>
      <td>130.6500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991</td>
      <td>144.800</td>
      <td>137.000</td>
      <td>134.100</td>
      <td>141.400</td>
      <td>137.900</td>
      <td>136.2000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1992</td>
      <td>150.000</td>
      <td>141.100</td>
      <td>139.000</td>
      <td>146.500</td>
      <td>142.500</td>
      <td>140.3000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1993</td>
      <td>154.500</td>
      <td>145.400</td>
      <td>142.900</td>
      <td>150.300</td>
      <td>146.300</td>
      <td>144.5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1994</td>
      <td>158.200</td>
      <td>148.600</td>
      <td>147.800</td>
      <td>152.300</td>
      <td>148.700</td>
      <td>148.2500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1995</td>
      <td>162.200</td>
      <td>153.300</td>
      <td>152.300</td>
      <td>154.600</td>
      <td>151.600</td>
      <td>152.3500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>166.900</td>
      <td>157.400</td>
      <td>157.500</td>
      <td>157.500</td>
      <td>155.100</td>
      <td>156.8500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1997</td>
      <td>170.800</td>
      <td>161.700</td>
      <td>163.000</td>
      <td>160.000</td>
      <td>160.400</td>
      <td>160.5500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1998</td>
      <td>173.600</td>
      <td>165.000</td>
      <td>167.700</td>
      <td>162.300</td>
      <td>165.500</td>
      <td>163.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1999</td>
      <td>177.000</td>
      <td>168.400</td>
      <td>172.800</td>
      <td>166.100</td>
      <td>172.500</td>
      <td>166.6000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>182.500</td>
      <td>173.800</td>
      <td>179.200</td>
      <td>171.600</td>
      <td>180.200</td>
      <td>172.2000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2001</td>
      <td>187.100</td>
      <td>178.300</td>
      <td>185.700</td>
      <td>177.300</td>
      <td>189.900</td>
      <td>177.0500</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2002</td>
      <td>191.900</td>
      <td>181.200</td>
      <td>189.300</td>
      <td>182.200</td>
      <td>193.000</td>
      <td>179.9000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2003</td>
      <td>197.800</td>
      <td>184.500</td>
      <td>192.300</td>
      <td>187.000</td>
      <td>196.400</td>
      <td>183.9500</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2004</td>
      <td>204.800</td>
      <td>188.600</td>
      <td>194.700</td>
      <td>193.200</td>
      <td>198.800</td>
      <td>188.9000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2005</td>
      <td>212.700</td>
      <td>194.300</td>
      <td>200.200</td>
      <td>201.800</td>
      <td>202.700</td>
      <td>195.3000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2006</td>
      <td>220.700</td>
      <td>198.300</td>
      <td>207.600</td>
      <td>210.400</td>
      <td>209.200</td>
      <td>201.6000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2007</td>
      <td>226.940</td>
      <td>204.818</td>
      <td>215.656</td>
      <td>217.338</td>
      <td>216.048</td>
      <td>207.3425</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2008</td>
      <td>235.782</td>
      <td>212.536</td>
      <td>224.719</td>
      <td>225.008</td>
      <td>222.767</td>
      <td>215.3030</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2009</td>
      <td>236.825</td>
      <td>209.995</td>
      <td>226.028</td>
      <td>223.219</td>
      <td>224.395</td>
      <td>214.5370</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2010</td>
      <td>240.864</td>
      <td>212.870</td>
      <td>226.693</td>
      <td>225.894</td>
      <td>227.469</td>
      <td>218.0555</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2011</td>
      <td>247.718</td>
      <td>218.684</td>
      <td>232.765</td>
      <td>231.928</td>
      <td>233.390</td>
      <td>224.9390</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2012</td>
      <td>252.588</td>
      <td>222.005</td>
      <td>238.663</td>
      <td>236.648</td>
      <td>239.650</td>
      <td>229.5940</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2013</td>
      <td>256.833</td>
      <td>224.545</td>
      <td>241.563</td>
      <td>239.207</td>
      <td>245.023</td>
      <td>232.9570</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2014</td>
      <td>260.230</td>
      <td>228.468</td>
      <td>246.018</td>
      <td>242.434</td>
      <td>251.985</td>
      <td>236.7360</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2015</td>
      <td>260.558</td>
      <td>227.792</td>
      <td>249.364</td>
      <td>244.632</td>
      <td>258.572</td>
      <td>237.0170</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2016</td>
      <td>263.365</td>
      <td>229.302</td>
      <td>254.886</td>
      <td>249.246</td>
      <td>266.344</td>
      <td>240.0075</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2017</td>
      <td>268.520</td>
      <td>233.611</td>
      <td>262.668</td>
      <td>256.210</td>
      <td>274.924</td>
      <td>245.1195</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2018</td>
      <td>273.641</td>
      <td>237.706</td>
      <td>271.089</td>
      <td>265.962</td>
      <td>285.550</td>
      <td>251.1070</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2019</td>
      <td>278.164</td>
      <td>241.181</td>
      <td>277.984</td>
      <td>274.114</td>
      <td>295.004</td>
      <td>255.6575</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2020</td>
      <td>282.920</td>
      <td>243.873</td>
      <td>282.693</td>
      <td>278.567</td>
      <td>300.084</td>
      <td>258.8110</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2021</td>
      <td>292.303</td>
      <td>254.159</td>
      <td>295.560</td>
      <td>289.244</td>
      <td>309.721</td>
      <td>270.9695</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Merge to df
df = pd.melt(cpi, id_vars=['year'], value_vars=cpi.columns[1:], 
            var_name='area', value_name='cpi') 

print(len(df)) # should be 32 years x 6 area = 192 rows
display(df.sample(5))
```

    192



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>area</th>
      <th>cpi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>2000</td>
      <td>SanFran</td>
      <td>180.200</td>
    </tr>
    <tr>
      <th>165</th>
      <td>1995</td>
      <td>All</td>
      <td>152.350</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2015</td>
      <td>Seattle</td>
      <td>249.364</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2011</td>
      <td>Seattle</td>
      <td>232.765</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1993</td>
      <td>Chicago</td>
      <td>145.400</td>
    </tr>
  </tbody>
</table>
</div>


## Cost of Living Index (COLI)
CPI may not account for actual living cost.
The COLI is closer to actual spend of living.
When we judge income level, we always consider the COLI to scale.

Unlike the CPI, finding COLI was difficult.
I found the COLI data of year 2010 from the Census.gov.
I will <font color=red>assume this number is staying same over years up to relative between cities.</font>

Also, it is <font color=red>not clear if this is calculated for metropolitan statistical area (broader) or only for city (smaller area). I'll assume the former, which is as same as the area division of wage statistics.</font>


```python
coli = pd.read_csv('data/COLI.csv')
coli = coli[coli.area!='Percent']
display(coli)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>coli</th>
      <th>Grocery</th>
      <th>Housing</th>
      <th>Utilities</th>
      <th>Transportation</th>
      <th>HealthCare</th>
      <th>Etc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NY</td>
      <td>216.7</td>
      <td>154.3</td>
      <td>386.7</td>
      <td>169.6</td>
      <td>120.3</td>
      <td>130.2</td>
      <td>145.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chicago</td>
      <td>116.9</td>
      <td>111.2</td>
      <td>134.8</td>
      <td>117.3</td>
      <td>116.5</td>
      <td>108.5</td>
      <td>104.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seattle</td>
      <td>121.4</td>
      <td>115.1</td>
      <td>140.3</td>
      <td>85.7</td>
      <td>118.8</td>
      <td>119.9</td>
      <td>119.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LA</td>
      <td>136.4</td>
      <td>106.0</td>
      <td>207.1</td>
      <td>101.7</td>
      <td>113.6</td>
      <td>109.1</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SanFran</td>
      <td>164.0</td>
      <td>111.9</td>
      <td>281.0</td>
      <td>94.5</td>
      <td>113.0</td>
      <td>117.0</td>
      <td>124.3</td>
    </tr>
  </tbody>
</table>
</div>


Housing price is the dominant driving factor.


```python
# Merge to df
df = df.merge(coli[['area','coli']], how = 'outer', on = ['area'])

print(len(df)) # should be 32 years x 6 area = 192 rows
display(df.sample(5))
```

    192



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>area</th>
      <th>cpi</th>
      <th>coli</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>2009</td>
      <td>SanFran</td>
      <td>224.395</td>
      <td>164.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1998</td>
      <td>Seattle</td>
      <td>167.700</td>
      <td>121.4</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2019</td>
      <td>Chicago</td>
      <td>241.181</td>
      <td>116.9</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1995</td>
      <td>Chicago</td>
      <td>153.300</td>
      <td>116.9</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2013</td>
      <td>LA</td>
      <td>239.207</td>
      <td>136.4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# COLI adjusted CPI (2010)
t = df[df.year==2010]
t['coli_cpi'] = t.coli/t.cpi
t = t[['area','coli_cpi']]
df = df.merge(t,how='left',on=['area'])
```

    /var/folders/31/7v9nfdf14sz0sxn2xwnq90y00000gn/T/ipykernel_12606/1860590002.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      t['coli_cpi'] = t.coli/t.cpi


## Census - U.S. and Metropolitan area population

- Source
    - National population
        - [Census.gov, 2010-2021](https://data.census.gov/cedsci/table?q=Age%20and%20Sex&tid=ACSDP1Y2010.DP05)
        - [Wikipedia, 1998-2009](https://en.wikipedia.org/wiki/Demographics_of_the_United_States)
    - Metropolitan statistical area population
        - [Census.gov, 2010-2021](https://www.census.gov/data/tables/time-series/demo/popest/2010s-total-metro-and-micro-statistical-areas.html)

### Metropolitan statistical area population data


```python
# Check metropolitan area census dataset
census = pd.read_csv('data/census.csv')
census.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2797 entries, 0 to 2796
    Data columns (total 19 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   CBSA               2797 non-null   int64  
     1   MDIV               141 non-null    float64
     2   STCOU              1840 non-null   float64
     3   NAME               2797 non-null   object 
     4   LSAD               2797 non-null   object 
     5   CENSUS2010POP      2797 non-null   int64  
     6   ESTIMATESBASE2010  2797 non-null   int64  
     7   POPESTIMATE2010    2797 non-null   int64  
     8   POPESTIMATE2011    2797 non-null   int64  
     9   POPESTIMATE2012    2797 non-null   int64  
     10  POPESTIMATE2013    2797 non-null   int64  
     11  POPESTIMATE2014    2797 non-null   int64  
     12  POPESTIMATE2015    2797 non-null   int64  
     13  POPESTIMATE2016    2797 non-null   int64  
     14  POPESTIMATE2017    2797 non-null   int64  
     15  POPESTIMATE2018    2797 non-null   int64  
     16  POPESTIMATE2019    2797 non-null   int64  
     17  POPESTIMATE2020    5 non-null      float64
     18  POPESTIMATE2021    5 non-null      float64
    dtypes: float64(4), int64(13), object(2)
    memory usage: 415.3+ KB



```python
# Select only interesting area

# Zipcode of Metropolitan Statistical Area
# LA has two zip codes because it has changed over years
zipcode_area = {31100:'LA',31080:'LA',41860:'SanFran',16980:'Chicago',35620:'NY',42660:'Seattle'}

lst=[]
for i in zipcode_area.keys():
    if i==31100:
        continue
    lst.append(census.loc[(census.CBSA==i)&(census.LSAD=='Metropolitan Statistical Area')])

census = pd.concat(lst)

display(census)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CBSA</th>
      <th>MDIV</th>
      <th>STCOU</th>
      <th>NAME</th>
      <th>LSAD</th>
      <th>CENSUS2010POP</th>
      <th>ESTIMATESBASE2010</th>
      <th>POPESTIMATE2010</th>
      <th>POPESTIMATE2011</th>
      <th>POPESTIMATE2012</th>
      <th>POPESTIMATE2013</th>
      <th>POPESTIMATE2014</th>
      <th>POPESTIMATE2015</th>
      <th>POPESTIMATE2016</th>
      <th>POPESTIMATE2017</th>
      <th>POPESTIMATE2018</th>
      <th>POPESTIMATE2019</th>
      <th>POPESTIMATE2020</th>
      <th>POPESTIMATE2021</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>850</th>
      <td>31080</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>Metropolitan Statistical Area</td>
      <td>12828837</td>
      <td>12828957</td>
      <td>12838417</td>
      <td>12925753</td>
      <td>13013443</td>
      <td>13097434</td>
      <td>13166609</td>
      <td>13234696</td>
      <td>13270694</td>
      <td>13278000</td>
      <td>13249879</td>
      <td>13214799</td>
      <td>13173266.0</td>
      <td>12997353.0</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>41860</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>San Francisco-Oakland-Berkeley, CA</td>
      <td>Metropolitan Statistical Area</td>
      <td>4335391</td>
      <td>4335593</td>
      <td>4343634</td>
      <td>4395725</td>
      <td>4455473</td>
      <td>4519636</td>
      <td>4584981</td>
      <td>4647924</td>
      <td>4688198</td>
      <td>4712421</td>
      <td>4726314</td>
      <td>4731803</td>
      <td>4739649.0</td>
      <td>4623264.0</td>
    </tr>
    <tr>
      <th>291</th>
      <td>16980</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Chicago-Naperville-Elgin, IL-IN-WI</td>
      <td>Metropolitan Statistical Area</td>
      <td>9461105</td>
      <td>9461537</td>
      <td>9470634</td>
      <td>9500870</td>
      <td>9528090</td>
      <td>9550194</td>
      <td>9560430</td>
      <td>9552554</td>
      <td>9533662</td>
      <td>9514113</td>
      <td>9484158</td>
      <td>9458539</td>
      <td>9601605.0</td>
      <td>9509934.0</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>35620</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New York-Newark-Jersey City, NY-NJ-PA</td>
      <td>Metropolitan Statistical Area</td>
      <td>18897109</td>
      <td>18896277</td>
      <td>18923407</td>
      <td>19052774</td>
      <td>19149689</td>
      <td>19226449</td>
      <td>19280929</td>
      <td>19320968</td>
      <td>19334778</td>
      <td>19322607</td>
      <td>19276644</td>
      <td>19216182</td>
      <td>20096413.0</td>
      <td>19768458.0</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>42660</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Seattle-Tacoma-Bellevue, WA</td>
      <td>Metropolitan Statistical Area</td>
      <td>3439809</td>
      <td>3439808</td>
      <td>3449241</td>
      <td>3503891</td>
      <td>3558829</td>
      <td>3612347</td>
      <td>3675160</td>
      <td>3739654</td>
      <td>3816355</td>
      <td>3885579</td>
      <td>3935179</td>
      <td>3979845</td>
      <td>4024730.0</td>
      <td>4011553.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Select only interesting fields
census.drop(['MDIV','STCOU','LSAD','CENSUS2010POP','ESTIMATESBASE2010'],axis=1,inplace=True)

# Change name of dields
census.columns = ['CBSA','NAME',2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]

# Transpose
census = census.T

# Put area names as field names
census.columns = ['LA','SanFran','Chicago','NY','Seattle']

# Reset index
census.reset_index()

# Leave only yearly population rows
census.drop(['CBSA','NAME'],inplace=True)

display(census)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LA</th>
      <th>SanFran</th>
      <th>Chicago</th>
      <th>NY</th>
      <th>Seattle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010</th>
      <td>12838417</td>
      <td>4343634</td>
      <td>9470634</td>
      <td>18923407</td>
      <td>3449241</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>12925753</td>
      <td>4395725</td>
      <td>9500870</td>
      <td>19052774</td>
      <td>3503891</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>13013443</td>
      <td>4455473</td>
      <td>9528090</td>
      <td>19149689</td>
      <td>3558829</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>13097434</td>
      <td>4519636</td>
      <td>9550194</td>
      <td>19226449</td>
      <td>3612347</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>13166609</td>
      <td>4584981</td>
      <td>9560430</td>
      <td>19280929</td>
      <td>3675160</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>13234696</td>
      <td>4647924</td>
      <td>9552554</td>
      <td>19320968</td>
      <td>3739654</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>13270694</td>
      <td>4688198</td>
      <td>9533662</td>
      <td>19334778</td>
      <td>3816355</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>13278000</td>
      <td>4712421</td>
      <td>9514113</td>
      <td>19322607</td>
      <td>3885579</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>13249879</td>
      <td>4726314</td>
      <td>9484158</td>
      <td>19276644</td>
      <td>3935179</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>13214799</td>
      <td>4731803</td>
      <td>9458539</td>
      <td>19216182</td>
      <td>3979845</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>13173266.0</td>
      <td>4739649.0</td>
      <td>9601605.0</td>
      <td>20096413.0</td>
      <td>4024730.0</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>12997353.0</td>
      <td>4623264.0</td>
      <td>9509934.0</td>
      <td>19768458.0</td>
      <td>4011553.0</td>
    </tr>
  </tbody>
</table>
</div>


### National population data


```python
# Check one example file of national population
demo = pd.read_csv('data/C2015.csv')

demo.info()
display(demo.head(5))
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 89 entries, 0 to 88
    Data columns (total 5 columns):
     #   Column                                  Non-Null Count  Dtype 
    ---  ------                                  --------------  ----- 
     0   Label (Grouping)                        89 non-null     object
     1   United States!!Estimate                 84 non-null     object
     2   United States!!Margin of Error          84 non-null     object
     3   United States!!Percent                  84 non-null     object
     4   United States!!Percent Margin of Error  84 non-null     object
    dtypes: object(5)
    memory usage: 3.6+ KB



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label (Grouping)</th>
      <th>United States!!Estimate</th>
      <th>United States!!Margin of Error</th>
      <th>United States!!Percent</th>
      <th>United States!!Percent Margin of Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SEX AND AGE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total population</td>
      <td>321,418,821</td>
      <td>*****</td>
      <td>321,418,821</td>
      <td>(X)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>158,167,834</td>
      <td>±31,499</td>
      <td>49.2%</td>
      <td>±0.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>163,250,987</td>
      <td>±31,500</td>
      <td>50.8%</td>
      <td>±0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Under 5 years</td>
      <td>19,793,807</td>
      <td>±16,520</td>
      <td>6.2%</td>
      <td>±0.1</td>
    </tr>
  </tbody>
</table>
</div>



```python
%%script false --no-raise-error
# If you already have data/usDemo.csv, this block can be skipped.

# Combine multiple year files of national population
df_save = []
for year in range(2010,2020):
    demo = pd.read_csv('data/C{0}.csv'.format(year))
    demo['year'] = year
    df_save.append(demo)
    
# Mave a csv file
df_save = pd.concat(df_save)

df_save.columns = ['label','estimate','estimate_err','pct','pct_err','year','estimate_err2']
df_save.to_csv('data/usDemo.csv', index=False)
```


```python
census_national = pd.read_csv('data/usDemo.csv')
display(census_national)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>estimate</th>
      <th>estimate_err</th>
      <th>pct</th>
      <th>pct_err</th>
      <th>year</th>
      <th>estimate_err2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SEX AND AGE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total population</td>
      <td>309,349,689</td>
      <td>*****</td>
      <td>309,349,689</td>
      <td>(X)</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>152,089,450</td>
      <td>±27,325</td>
      <td>49.2%</td>
      <td>±0.1</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>157,260,239</td>
      <td>±27,325</td>
      <td>50.8%</td>
      <td>±0.1</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Under 5 years</td>
      <td>20,133,943</td>
      <td>±20,568</td>
      <td>6.5%</td>
      <td>±0.1</td>
      <td>2010</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>880</th>
      <td>Total housing units</td>
      <td>139,686,209</td>
      <td>NaN</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>2019</td>
      <td>±6,973</td>
    </tr>
    <tr>
      <th>881</th>
      <td>CITIZEN, VOTING AGE POPULATION</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>882</th>
      <td>Citizen, 18 and over population</td>
      <td>235,418,734</td>
      <td>NaN</td>
      <td>235,418,734</td>
      <td>(X)</td>
      <td>2019</td>
      <td>±159,764</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Male</td>
      <td>114,206,194</td>
      <td>NaN</td>
      <td>48.5%</td>
      <td>±0.1</td>
      <td>2019</td>
      <td>±98,225</td>
    </tr>
    <tr>
      <th>884</th>
      <td>Female</td>
      <td>121,212,540</td>
      <td>NaN</td>
      <td>51.5%</td>
      <td>±0.1</td>
      <td>2019</td>
      <td>±79,689</td>
    </tr>
  </tbody>
</table>
<p>885 rows × 7 columns</p>
</div>



```python
# Leave data to use only
# Change format to merge with metropolitan census dataframe

population=[]
years=[]

# make year:population dictionary
for year in range(2010,2020):
    pop = int(''.join(census_national[(census_national.label.str.contains('Total population'))\
                                      &(census_national.year==year)].iloc[0].estimate.split(',')))
    population.append(pop)
    years.append(year)
    

census_national = pd.DataFrame({'year':years,'All':population})
```


```python
# Add extra years

census_national_other = pd.DataFrame([
    [2020, 331501080],
    [2021, 331893745],
    [1998, 275854000],
    [1999, 279040000],
    [2000, 282172000],
    [2001, 285082000],
    [2002, 287804000],
    [2003, 290326000],
    [2004, 293046000],
    [2005, 295753000],
    [2006, 298593000],
    [2007, 301580000],
    [2008, 304375000],
    [2009, 307007000]], 
    columns=['year','All'])


census_national = census_national.append(census_national_other, ignore_index=True)


census_national = census_national.sort_values(by=['year'])

display(census_national)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>All</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1998</td>
      <td>275854000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1999</td>
      <td>279040000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2000</td>
      <td>282172000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2001</td>
      <td>285082000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2002</td>
      <td>287804000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2003</td>
      <td>290326000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004</td>
      <td>293046000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2005</td>
      <td>295753000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006</td>
      <td>298593000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2007</td>
      <td>301580000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2008</td>
      <td>304375000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2009</td>
      <td>307007000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>309349689</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>311591919</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>313914040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>316128839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>318857056</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>321418821</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016</td>
      <td>323127515</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017</td>
      <td>325719178</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>327167439</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019</td>
      <td>328239523</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020</td>
      <td>331501080</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2021</td>
      <td>331893745</td>
    </tr>
  </tbody>
</table>
</div>


### Merge national population to metropolitan population


```python
census = pd.merge(census_national, census, how='outer', right_index=True, left_on='year')

# Change data type to numeric
census = census.apply(pd.to_numeric)

census.info()
display(census)
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 24 entries, 12 to 11
    Data columns (total 7 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   year     24 non-null     int64  
     1   All      24 non-null     int64  
     2   LA       12 non-null     float64
     3   SanFran  12 non-null     float64
     4   Chicago  12 non-null     float64
     5   NY       12 non-null     float64
     6   Seattle  12 non-null     float64
    dtypes: float64(5), int64(2)
    memory usage: 1.5 KB



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>All</th>
      <th>LA</th>
      <th>SanFran</th>
      <th>Chicago</th>
      <th>NY</th>
      <th>Seattle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1998</td>
      <td>275854000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1999</td>
      <td>279040000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2000</td>
      <td>282172000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2001</td>
      <td>285082000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2002</td>
      <td>287804000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2003</td>
      <td>290326000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004</td>
      <td>293046000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2005</td>
      <td>295753000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2006</td>
      <td>298593000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2007</td>
      <td>301580000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2008</td>
      <td>304375000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2009</td>
      <td>307007000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2010</td>
      <td>309349689</td>
      <td>12838417.0</td>
      <td>4343634.0</td>
      <td>9470634.0</td>
      <td>18923407.0</td>
      <td>3449241.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>311591919</td>
      <td>12925753.0</td>
      <td>4395725.0</td>
      <td>9500870.0</td>
      <td>19052774.0</td>
      <td>3503891.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012</td>
      <td>313914040</td>
      <td>13013443.0</td>
      <td>4455473.0</td>
      <td>9528090.0</td>
      <td>19149689.0</td>
      <td>3558829.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>316128839</td>
      <td>13097434.0</td>
      <td>4519636.0</td>
      <td>9550194.0</td>
      <td>19226449.0</td>
      <td>3612347.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>318857056</td>
      <td>13166609.0</td>
      <td>4584981.0</td>
      <td>9560430.0</td>
      <td>19280929.0</td>
      <td>3675160.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>321418821</td>
      <td>13234696.0</td>
      <td>4647924.0</td>
      <td>9552554.0</td>
      <td>19320968.0</td>
      <td>3739654.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016</td>
      <td>323127515</td>
      <td>13270694.0</td>
      <td>4688198.0</td>
      <td>9533662.0</td>
      <td>19334778.0</td>
      <td>3816355.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017</td>
      <td>325719178</td>
      <td>13278000.0</td>
      <td>4712421.0</td>
      <td>9514113.0</td>
      <td>19322607.0</td>
      <td>3885579.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>327167439</td>
      <td>13249879.0</td>
      <td>4726314.0</td>
      <td>9484158.0</td>
      <td>19276644.0</td>
      <td>3935179.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019</td>
      <td>328239523</td>
      <td>13214799.0</td>
      <td>4731803.0</td>
      <td>9458539.0</td>
      <td>19216182.0</td>
      <td>3979845.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020</td>
      <td>331501080</td>
      <td>13173266.0</td>
      <td>4739649.0</td>
      <td>9601605.0</td>
      <td>20096413.0</td>
      <td>4024730.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2021</td>
      <td>331893745</td>
      <td>12997353.0</td>
      <td>4623264.0</td>
      <td>9509934.0</td>
      <td>19768458.0</td>
      <td>4011553.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Merge to df
census = pd.melt(census, id_vars=['year'], value_vars=cpi.columns[1:], 
        var_name='area', value_name='population')

df = df.merge(census, how='outer', on= ['area','year'])

print(len(df)) # should be 32 years x 6 area = 192 rows
display(df.sample(5))
```

    192



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>area</th>
      <th>cpi</th>
      <th>coli</th>
      <th>coli_cpi</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>2011</td>
      <td>Chicago</td>
      <td>218.6840</td>
      <td>116.9</td>
      <td>0.549161</td>
      <td>9500870.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2019</td>
      <td>Chicago</td>
      <td>241.1810</td>
      <td>116.9</td>
      <td>0.549161</td>
      <td>9458539.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2000</td>
      <td>Chicago</td>
      <td>173.8000</td>
      <td>116.9</td>
      <td>0.549161</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2010</td>
      <td>All</td>
      <td>218.0555</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>309349689.0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2017</td>
      <td>Seattle</td>
      <td>262.6680</td>
      <td>121.4</td>
      <td>0.535526</td>
      <td>3885579.0</td>
    </tr>
  </tbody>
</table>
</div>


# U.S. market data

I crated "data/rev.csv" file by combining U.S. markets statistics from multiple sources.
Data sources are
- U.S. dance studio 
    - market size: [Statista](https://www.statista.com/statistics/1175824/dance-studio-industry-market-size-us/)
    - number of businesses: [IBISWorld](https://www.ibisworld.com/industry-statistics/number-of-businesses/dance-studios-united-states/)
    - number of employees: [IBISWorld](https://www.ibisworld.com/industry-statistics/employment/dance-studios-united-states/)
    - wages: [IBISWorld](https://www.ibisworld.com/industry-statistics/wages/dance-studios-united-states/)
- U.S. fitness and recreational sports centers 
    - revenue: [FRED, cited U.S. Bureau of Labor Statistics](https://fred.stlouisfed.org/series/REVEF71394ALLEST).


```python
# Read data file
rev = pd.read_csv('data/rev.csv')

# Set every money scale to billion dollars
rev.fitness = rev.fitness/1000
rev.studio_wage = rev.studio_wage/1000

display(rev)
# fitness: U.S. fitness and recreational sports center revenue
# dance_studio: U.S. dance studio revenue
# studio_num: number of buinesses of U.S. dance studio
# studio_emp: number of employees of U.S. dance studio
# studio_wage: total wage of U.S. dance studio
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>fitness</th>
      <th>dance_studio</th>
      <th>studio_num</th>
      <th>studio_emp</th>
      <th>studio_wage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1998</td>
      <td>10.797</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1999</td>
      <td>11.777</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>12.543</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>13.542</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>14.987</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2003</td>
      <td>16.287</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004</td>
      <td>17.174</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2005</td>
      <td>18.286</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2006</td>
      <td>19.447</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2007</td>
      <td>21.416</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2008</td>
      <td>22.339</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2009</td>
      <td>21.842</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2010</td>
      <td>22.311</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2011</td>
      <td>23.191</td>
      <td>3.04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2012</td>
      <td>24.051</td>
      <td>3.22</td>
      <td>47269.0</td>
      <td>90668.0</td>
      <td>0.9026</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2013</td>
      <td>25.803</td>
      <td>3.28</td>
      <td>48399.0</td>
      <td>93420.0</td>
      <td>0.9029</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014</td>
      <td>27.001</td>
      <td>3.42</td>
      <td>52942.0</td>
      <td>99696.0</td>
      <td>0.9504</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015</td>
      <td>28.838</td>
      <td>3.59</td>
      <td>55523.0</td>
      <td>104321.0</td>
      <td>1.0148</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2016</td>
      <td>31.223</td>
      <td>3.70</td>
      <td>56412.0</td>
      <td>107832.0</td>
      <td>1.0798</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2017</td>
      <td>33.042</td>
      <td>3.87</td>
      <td>58515.0</td>
      <td>114075.0</td>
      <td>1.1448</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018</td>
      <td>33.971</td>
      <td>4.10</td>
      <td>63363.0</td>
      <td>120456.0</td>
      <td>1.1787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2019</td>
      <td>35.889</td>
      <td>4.20</td>
      <td>65723.0</td>
      <td>126288.0</td>
      <td>1.2295</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>24.361</td>
      <td>3.43</td>
      <td>62808.0</td>
      <td>112485.0</td>
      <td>1.0768</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2021</td>
      <td>NaN</td>
      <td>3.72</td>
      <td>66266.0</td>
      <td>120081.0</td>
      <td>1.1531</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2022</td>
      <td>NaN</td>
      <td>3.83</td>
      <td>68393.0</td>
      <td>123680.0</td>
      <td>1.1876</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Merge to df
df= df.merge(rev, how = 'outer', on= ['year'])

# for convenience
df.loc[df.area.isna(),'area']='All'

print(len(df)) # should be 32 years x 6 area + 1 year row = 193 rows
display(df.sample(3))
```

    193



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>area</th>
      <th>cpi</th>
      <th>coli</th>
      <th>coli_cpi</th>
      <th>population</th>
      <th>fitness</th>
      <th>dance_studio</th>
      <th>studio_num</th>
      <th>studio_emp</th>
      <th>studio_wage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>2013</td>
      <td>NY</td>
      <td>256.833</td>
      <td>216.7</td>
      <td>0.899678</td>
      <td>19226449.0</td>
      <td>25.803</td>
      <td>3.28</td>
      <td>48399.0</td>
      <td>93420.0</td>
      <td>0.9029</td>
    </tr>
    <tr>
      <th>115</th>
      <td>2009</td>
      <td>Chicago</td>
      <td>209.995</td>
      <td>116.9</td>
      <td>0.549161</td>
      <td>NaN</td>
      <td>21.842</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2007</td>
      <td>NY</td>
      <td>226.940</td>
      <td>216.7</td>
      <td>0.899678</td>
      <td>NaN</td>
      <td>21.416</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


# Employee statistics data
- Source: [Occupational Employment and Wage Statistics provided by U.S. Bureau of Labor Statistics](https://www.bls.gov/oes/tables.htm).

This website provides a table of employment statistics (wage, number of employee, etc) of different area of each year.
I downloaded each year's file, and they will be cleaned and concatenated.

## Explanation of fields

Here are definitions of each field. Not explained field is not used in this analysis.

### Area identifier
- area: area code 
- area_name(title):	Area name 

### Job identifier
- occ_code: The 6-digit Standard Occupational Classification (SOC) code or OEWS-specific code for the occupation 
- occ_title: SOC title or OEWS-specific title for the occupation

### Number of employee
- tot_emp: Estimated total employment rounded to the nearest 10 (excludes self-employed).
- emp_prse:	Percent relative standard error (PRSE) for the employment estimate. PRSE is a measure of sampling error, expressed as a percentage of the corresponding estimate. Sampling error occurs when values for a population are estimated from a sample survey of the population, rather than calculated from data for all members of the population. Estimates with lower PRSEs are typically more precise in the presence of sampling error.

### Wage
- h_mean: Mean hourly wage
- a_mean: Mean annual wage 

- mean_prse: Percent relative standard error (PRSE) for the mean wage estimate.
- h_pct10: Hourly 10th percentile wage
- h_pct25: Hourly 25th percentile wage
- h_median: Hourly median wage (or the 50th percentile)
- h_pct75: Hourly 75th percentile wage
- h_pct90: Hourly 90th percentile wage

- a_pct10: Annual 10th percentile wage
- a_pct25: Annual 25th percentile wage
- a_median: Annual median wage (or the 50th percentile)
- a_pct75: Annual 75th percentile wage
- a_pct90: Annual 90th percentile wage


```python
# Let's check how each file looks like
sample = pd.read_excel('data/2010.xls')

sample.info()
display(sample.sample(2))
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7940 entries, 0 to 7939
    Data columns (total 25 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   PRIM_STATE    7940 non-null   object
     1   AREA          7940 non-null   int64 
     2   AREA_NAME     7940 non-null   object
     3   OCC_CODE      7940 non-null   object
     4   OCC_TITLE     7940 non-null   object
     5   GROUP         253 non-null    object
     6   TOT_EMP       7940 non-null   object
     7   EMP_PRSE      7940 non-null   object
     8   JOBS_1000     7940 non-null   object
     9   LOC QUOTIENT  7940 non-null   object
     10  H_MEAN        7940 non-null   object
     11  A_MEAN        7940 non-null   object
     12  MEAN_PRSE     7940 non-null   object
     13  H_PCT10       7940 non-null   object
     14  H_PCT25       7940 non-null   object
     15  H_MEDIAN      7940 non-null   object
     16  H_PCT75       7940 non-null   object
     17  H_PCT90       7940 non-null   object
     18  A_PCT10       7940 non-null   object
     19  A_PCT25       7940 non-null   object
     20  A_MEDIAN      7940 non-null   object
     21  A_PCT75       7940 non-null   object
     22  A_PCT90       7940 non-null   object
     23  ANNUAL        562 non-null    object
     24  HOURLY        38 non-null     object
    dtypes: int64(1), object(24)
    memory usage: 1.5+ MB



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRIM_STATE</th>
      <th>AREA</th>
      <th>AREA_NAME</th>
      <th>OCC_CODE</th>
      <th>OCC_TITLE</th>
      <th>GROUP</th>
      <th>TOT_EMP</th>
      <th>EMP_PRSE</th>
      <th>JOBS_1000</th>
      <th>LOC QUOTIENT</th>
      <th>...</th>
      <th>H_MEDIAN</th>
      <th>H_PCT75</th>
      <th>H_PCT90</th>
      <th>A_PCT10</th>
      <th>A_PCT25</th>
      <th>A_MEDIAN</th>
      <th>A_PCT75</th>
      <th>A_PCT90</th>
      <th>ANNUAL</th>
      <th>HOURLY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3384</th>
      <td>IL</td>
      <td>16980</td>
      <td>Chicago-Naperville-Joliet, IL-IN-WI</td>
      <td>47-2041</td>
      <td>Carpet Installers</td>
      <td>NaN</td>
      <td>1470</td>
      <td>17.6</td>
      <td>0.352</td>
      <td>1.749</td>
      <td>...</td>
      <td>22.34</td>
      <td>32.03</td>
      <td>39.24</td>
      <td>23180</td>
      <td>35810</td>
      <td>46470</td>
      <td>66630</td>
      <td>81620</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2762</th>
      <td>FL</td>
      <td>33100</td>
      <td>Miami-Fort Lauderdale-Pompano Beach, FL</td>
      <td>51-4023</td>
      <td>Rolling Machine Setters, Operators, and Tender...</td>
      <td>NaN</td>
      <td>190</td>
      <td>16.4</td>
      <td>0.09</td>
      <td>0.359</td>
      <td>...</td>
      <td>15.46</td>
      <td>18.64</td>
      <td>20.53</td>
      <td>18690</td>
      <td>25990</td>
      <td>32160</td>
      <td>38780</td>
      <td>42700</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 25 columns</p>
</div>



```python
# jobs in interest
    # dancer: dancer
    # choreo: choreographer, 
    # fit_trainer: fitness trainer/instructer
    # rec_worker: recreational worker, 
    # all_jobs: all jobs sum/mean
    
code_job = {'27-2031':'dancer','27-2032':'choreo',
            '39-9031':'fit_trainer','39-9032':'rec_worker',
            '00-0000':'all_jobs'}
```


```python
%%script false --no-raise-error
# This block combines multiple wage data files, then generate a single csv file.
# If you already have data/wage.csv, this block can be skipped. It takes time to run.


df_save = []
for year in range(2003,2022):
    
    print(year)
    metro = None # metropolitan area statistics data
    national = None # national statistics data
    
    try:
        if year>2004:
            metro = pd.read_excel('data/'+str(year)+'.xls')
        national = pd.read_excel('data/'+str(year)+'nat.xls')
    except:
        if year>2004:
            metro = pd.read_excel('data/'+str(year)+'.xlsx')
        national = pd.read_excel('data/'+str(year)+'nat.xlsx')

    if year>2004:
        metro.columns = metro.columns.str.strip().str.lower()
    national.columns = national.columns.str.strip().str.lower()
    
    # unify feature names in all years
    if year>2004:
        metro.rename(columns={'area_title':'area_name'},inplace=True)
        
    # LA area code changed
    area_la = 31100
    if year>2014:
        area_la=31080

    if year>2004:
        # Select metropolitan area in interest
        metro = metro.loc[(metro.area==area_la) | (metro.area==41860) | (metro.area==16980) | 
                        (metro.area==35620) | (metro.area==42660)]

        # Select occupation in interest
        metro = metro.loc[(metro.occ_code=='27-2031') | (metro.occ_code=='27-2032') | 
                          (metro.occ_code=='39-9031') | (metro.occ_code=='39-9032') | 
                          (metro.occ_code=='00-0000')]
        # Change zip code to the unique area names
        metro['area']=metro.apply(lambda x: zipcode_area[x['area']], axis=1)
        
    # Select occupation in interest
    national = national.loc[(national.occ_code=='27-2031') | (national.occ_code=='27-2032') |
                            (national.occ_code=='39-9031') | (national.occ_code=='39-9032') |                          
                            (national.occ_code=='00-0000')]

    


    # To match columns with metropolitan dataframe
    national['area'] = 'All'
    national['area_name'] = 'U.S. all'
    
    # Keep only columns to use
    if year>2004:
        metro = metro[['area', 'area_name', 'occ_code', 'occ_title', 
           'tot_emp', 'emp_prse', 'h_mean', 'a_mean', 'mean_prse', 'h_pct10',
           'h_pct25', 'h_median', 'h_pct75', 'h_pct90', 'a_pct10', 'a_pct25',
           'a_median', 'a_pct75', 'a_pct90']]

    national = national[['area', 'area_name', 'occ_code', 'occ_title', 
   'tot_emp', 'emp_prse', 'h_mean', 'a_mean', 'mean_prse', 'h_pct10',
   'h_pct25', 'h_median', 'h_pct75', 'h_pct90', 'a_pct10', 'a_pct25',
   'a_median', 'a_pct75', 'a_pct90']]
    
    
    # comebine national data to metropolitan data
    emp=None
    
    if year>2004:
        emp = pd.concat([national,metro], ignore_index=True)
    else:
        emp = national

    # add year
    emp['year']=year

    # add the unique occupation name
    emp['occ']=emp.apply(lambda x: code_job[x['occ_code']], axis=1)

    # Cleaning
    emp.replace('**',np.nan,inplace=True)
    emp.replace('*',np.nan,inplace=True)

    # Append to a list to save
    df_save.append(emp)

# Mave a csv file
df_save = pd.concat(df_save)
df_save.to_csv('data/emp.csv', index=False)
```


```python
# Check data is prepared as intended
emp = pd.read_csv('data/emp.csv')

emp.info()

# Confirm if city and occupation labels are correct
print("Check area names are correctly marked ------ ")
for x in zipcode_area.values():
    print(x,emp[emp.area==x].area_name.unique())
    
print("\n Check occupation names are correctly marked ------ ")    
for x in code_job.values():
    print(x,emp[emp.occ==x].occ_title.unique())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 497 entries, 0 to 496
    Data columns (total 21 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   area       497 non-null    object 
     1   area_name  497 non-null    object 
     2   occ_code   497 non-null    object 
     3   occ_title  497 non-null    object 
     4   tot_emp    462 non-null    float64
     5   emp_prse   462 non-null    float64
     6   h_mean     491 non-null    float64
     7   a_mean     402 non-null    float64
     8   mean_prse  491 non-null    float64
     9   h_pct10    491 non-null    float64
     10  h_pct25    491 non-null    float64
     11  h_median   491 non-null    float64
     12  h_pct75    491 non-null    float64
     13  h_pct90    491 non-null    float64
     14  a_pct10    402 non-null    float64
     15  a_pct25    402 non-null    float64
     16  a_median   402 non-null    float64
     17  a_pct75    402 non-null    float64
     18  a_pct90    402 non-null    float64
     19  year       497 non-null    int64  
     20  occ        497 non-null    object 
    dtypes: float64(15), int64(1), object(5)
    memory usage: 81.7+ KB
    Check area names are correctly marked ------ 
    LA ['Los Angeles-Long Beach-Santa Ana, CA'
     'Los Angeles-Long Beach-Anaheim, CA']
    LA ['Los Angeles-Long Beach-Santa Ana, CA'
     'Los Angeles-Long Beach-Anaheim, CA']
    SanFran ['San Francisco-Oakland-Fremont, CA' 'San Francisco-Oakland-Hayward, CA']
    Chicago ['Chicago-Naperville-Joilet, IL-IN-WI'
     'Chicago-Naperville-Joliet, IL-IN-WI'
     'Chicago-Joliet-Naperville, IL-IN-WI'
     'Chicago-Naperville-Elgin, IL-IN-WI']
    NY ['New York-Northern New Jersey-Long Island, NY-NJ-PA'
     'New York-Newark-Jersey City, NY-NJ-PA']
    Seattle ['Seattle-Tacoma-Bellevue, WA']
    
     Check occupation names are correctly marked ------ 
    dancer ['Dancers']
    choreo ['Choreographers']
    fit_trainer ['Fitness trainers and aerobics instructors'
     'Fitness Trainers and Aerobics Instructors'
     'Exercise Trainers and Group Fitness Instructors']
    rec_worker ['Recreation workers' 'Recreation Workers']
    all_jobs ['All Occupations']



```python
# Since area and occupations are correctly marked, let's remove them
emp.drop(['area_name','occ_code','occ_title'],axis=1,inplace=True)
print(len(emp)) # 17 years x 6 area x 5 jobs + 2 years x 1 area x 5 jobs = 520 rows, if no missing record
display(emp.sample(5))
```

    497



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>tot_emp</th>
      <th>emp_prse</th>
      <th>h_mean</th>
      <th>a_mean</th>
      <th>mean_prse</th>
      <th>h_pct10</th>
      <th>h_pct25</th>
      <th>h_median</th>
      <th>h_pct75</th>
      <th>h_pct90</th>
      <th>a_pct10</th>
      <th>a_pct25</th>
      <th>a_median</th>
      <th>a_pct75</th>
      <th>a_pct90</th>
      <th>year</th>
      <th>occ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>385</th>
      <td>Seattle</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.82</td>
      <td>78670.0</td>
      <td>10.3</td>
      <td>20.78</td>
      <td>34.29</td>
      <td>40.35</td>
      <td>46.00</td>
      <td>49.36</td>
      <td>43220.0</td>
      <td>71320.0</td>
      <td>83920.0</td>
      <td>95690.0</td>
      <td>102660.0</td>
      <td>2017</td>
      <td>choreo</td>
    </tr>
    <tr>
      <th>410</th>
      <td>NY</td>
      <td>70.0</td>
      <td>33.8</td>
      <td>40.86</td>
      <td>84990.0</td>
      <td>12.8</td>
      <td>17.68</td>
      <td>21.40</td>
      <td>37.44</td>
      <td>59.55</td>
      <td>73.45</td>
      <td>36780.0</td>
      <td>44500.0</td>
      <td>77870.0</td>
      <td>123870.0</td>
      <td>152780.0</td>
      <td>2018</td>
      <td>choreo</td>
    </tr>
    <tr>
      <th>384</th>
      <td>Seattle</td>
      <td>110.0</td>
      <td>10.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017</td>
      <td>dancer</td>
    </tr>
    <tr>
      <th>166</th>
      <td>LA</td>
      <td>13510.0</td>
      <td>3.8</td>
      <td>12.09</td>
      <td>25150.0</td>
      <td>1.1</td>
      <td>8.76</td>
      <td>9.65</td>
      <td>11.32</td>
      <td>13.75</td>
      <td>16.49</td>
      <td>18210.0</td>
      <td>20060.0</td>
      <td>23550.0</td>
      <td>28590.0</td>
      <td>34290.0</td>
      <td>2010</td>
      <td>rec_worker</td>
    </tr>
    <tr>
      <th>291</th>
      <td>NY</td>
      <td>8615710.0</td>
      <td>0.3</td>
      <td>28.39</td>
      <td>59060.0</td>
      <td>0.5</td>
      <td>9.19</td>
      <td>12.47</td>
      <td>20.99</td>
      <td>35.83</td>
      <td>55.72</td>
      <td>19120.0</td>
      <td>25950.0</td>
      <td>43660.0</td>
      <td>74530.0</td>
      <td>115900.0</td>
      <td>2014</td>
      <td>all_jobs</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Merge to df
df= df.merge(emp, how = 'outer', on= ['area','year'])

print(len(df)) # 520 rows, if no missing record +
                # 89 rows, 15 years x 5 area + 14 years x 1 area = 609
display(df.sample(5))
```

    586



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>area</th>
      <th>cpi</th>
      <th>coli</th>
      <th>coli_cpi</th>
      <th>population</th>
      <th>fitness</th>
      <th>dance_studio</th>
      <th>studio_num</th>
      <th>studio_emp</th>
      <th>...</th>
      <th>h_pct25</th>
      <th>h_median</th>
      <th>h_pct75</th>
      <th>h_pct90</th>
      <th>a_pct10</th>
      <th>a_pct25</th>
      <th>a_median</th>
      <th>a_pct75</th>
      <th>a_pct90</th>
      <th>occ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>2000</td>
      <td>SanFran</td>
      <td>180.200</td>
      <td>164.0</td>
      <td>0.720977</td>
      <td>NaN</td>
      <td>12.543</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>520</th>
      <td>2019</td>
      <td>LA</td>
      <td>274.114</td>
      <td>136.4</td>
      <td>0.603823</td>
      <td>13214799.0</td>
      <td>35.889</td>
      <td>4.20</td>
      <td>65723.0</td>
      <td>126288.0</td>
      <td>...</td>
      <td>20.69</td>
      <td>34.81</td>
      <td>54.68</td>
      <td>59.31</td>
      <td>26400.0</td>
      <td>43020.0</td>
      <td>72400.0</td>
      <td>113740.0</td>
      <td>123350.0</td>
      <td>choreo</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2008</td>
      <td>LA</td>
      <td>225.008</td>
      <td>136.4</td>
      <td>0.603823</td>
      <td>NaN</td>
      <td>22.339</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>14.11</td>
      <td>21.63</td>
      <td>28.35</td>
      <td>33.29</td>
      <td>25440.0</td>
      <td>29340.0</td>
      <td>44980.0</td>
      <td>58970.0</td>
      <td>69240.0</td>
      <td>choreo</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1997</td>
      <td>NY</td>
      <td>170.800</td>
      <td>216.7</td>
      <td>0.899678</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>357</th>
      <td>2013</td>
      <td>All</td>
      <td>232.957</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>316128839.0</td>
      <td>25.803</td>
      <td>3.28</td>
      <td>48399.0</td>
      <td>93420.0</td>
      <td>...</td>
      <td>10.15</td>
      <td>15.88</td>
      <td>23.36</td>
      <td>32.19</td>
      <td>17840.0</td>
      <td>21110.0</td>
      <td>33020.0</td>
      <td>48590.0</td>
      <td>66950.0</td>
      <td>fit_trainer</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>


# Save organized dataset
Now, we have all data prepared. 
Let's save it for next steps.


```python
# for convenience
df.loc[df.occ.isna(),'occ']='all_jobs'
```


```python
#%%script false --no-raise-error
# If you already have data/dance.csv, this block can be skipped.
df.to_csv('data/dance1.csv',index=False)
```
