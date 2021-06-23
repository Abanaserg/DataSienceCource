import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import openpyxl
from typing import List, Dict, Optional
import numpy as np
import math

from itertools import combinations
from scipy.stats import ttest_ind
from scipy.stats import norm
from scipy.stats import t
from IPython.display import display
from statsmodels.stats import weightstats

# Вас пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в
# повышении уровня благополучия детей по всему миру.
#
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике,
# чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
#
# И сделать это можно с помощью модели, которая предсказывала бы результаты госэкзамена по математике для каждого
# ученика школы (вот она, сила ML!). Чтобы определиться с параметрами будущей модели, проведите разведывательный
# анализ данных и составьте отчёт по его результатам.

# Разведывательный анализ данных (EDA)
# Основные цели EDA:
#
# - Сформулировать предположения и гипотезы для дальнейшего построения модели.
# - Проверить качество данных и очистить их, если это необходимо.
# - Определиться с параметрами модели.
#
# Основные шаги:
#
# - Посмотреть на данные +
# - Проверить данные на пустые значения +
# - Проверить данные на дублированные/полностью скоррелированные значения.
# - Проверить данные на наличие выбросов +
# - Отобрать данные, пригодные для дальнейшего построения модели.
# - Снова и снова возвращаться к предыдущим пунктам, пока модель не заработает как надо.

# Dictionaries for next changes
school={'GP':0,'MS':1}
sex={'F':0,'M':1}
address={'U':0,'R':1}
family_size={'LE3':0,'GT3':1}
parent_status={'T':0,'A':1}
education={'No_educat':0,'4_classes':1,'5-9_classes':2,'midspec/11_class':3,'hight_educat':4} # desipher
job={'teacher':0, 'health':1, 'services':2, 'at_home':3, 'other':4}
reason={'home':0, 'reputation':1, 'course':2, 'other':3}
guardian={'mother':0, 'father':1, 'other':2}
travel_time={'<15 m':1,'15-30 m':2,'30-60 m':3,'>60 m':4}  # desipher
study_time={'<2 h':1,'2-5 h':2,'5-10 h':3,'>10 h':4}  # desipher
yes_no_colm={'no':0,'yes':1}

# Function for calculating percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'Q_%s' % (n//25)
    return percentile_


# Function for plotting the boxplots and table for 1 column
def obj_plots(column, df):
    # Create statistic table for column score groupped by testing column
    column_stats = df.groupby(by=column)['score'].agg(['count', 'mean', 'std', 'median',
                                                       'min', 'max', percentile(25), percentile(75)])
    # Remove values that numbers less than 5 (not significant)
    column_stats = column_stats[column_stats['count'] > 5]

    if len(column_stats) > 1: # Testing that we still have something to test)
        # Plot the pattern for boxplots and statistic table
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 1]})

        # Configure title and axes lables
        axes[0].set_title('Boxplot of scores by')
        axes[0].xaxis.set_label_position('top')
        axes[0].xaxis.tick_top()
        axes[1].axis('tight')
        axes[1].axis('off')

        # Find significant diference
        stat_result_2 = get_stat_dif(column, df)

        # Set table name
        axes[1].set_title(stat_result_2)

        # Plot the statistic table
        table = axes[1].table(cellText=np.round(column_stats.values.tolist(), 1),
                              colLabels=column_stats.columns,
                              rowLabels=column_stats.index)
        # Configure table parametrs
        table.auto_set_font_size(False)
        table.set_fontsize(14)

        # Plot the boxplot for scores, calculated by used column
        sns.boxplot(ax=axes[0], x=column,
                    y='score',
                    data=df)

        # Plot the figure
        #plt.show()

# Function for finding significant difference
def get_stat_dif(column_1, df):
    cols = df.loc[:, column_1].value_counts().index # Use unique values for next calculations
    combinations_all = list(combinations(cols, 2)) # Create combination table for finding significant difference

    for comb in combinations_all: # Testing all combinations

        # values of finding frame length
        n_comb_0 = len(df.loc[df.loc[:, column_1] == comb[0], 'score'])
        n_comb_1 = len(df.loc[df.loc[:, column_1] == comb[1], 'score'])

        # Test the data frame account for frames length
        if (n_comb_0 > 5 and n_comb_1 > 5) and (ttest_ind(df.loc[df.loc[:, column_1] == comb[0], 'score'],
                                                          df.loc[df.loc[:, column_1] == comb[1], 'score']).pvalue \
                                                <= 0.05 / len(combinations_all)):  # Учли поправку Бонферони
            # If data Frame have significant diference return informaniton about it
            return 'There are significant diference for colunm ' + column_1

    # If data Frame have no significant diference return informaniton about it
    return "There aren't significant diference for colunm " + column_1

# Function for turn NaN Values in object column into None
def nan_to_none(column):
    return pupil_math[column].apply(lambda x: None if pd.isnull(x) else x)

pd.set_option('display.max_rows', 50) # Show more rows
pd.set_option('display.max_columns', 50) # Show more columns

pupil_math=pd.read_csv('../stud_math.csv') #File import

display(pupil_math.columns) # Show original columns names
display(pupil_math.head(10)) # Show first 10 rows
pupil_math.info() # Show DS info

# Change columns names for next analysis
pupil_math.columns=['school', 'sex', 'age', 'address', 'family_size', 'parent_status', 'mother_education',
                    'father_education', 'mother_job', 'father_job', 'reason', 'guardian', 'travel_time', 'study_time',
                    'failures', 'schools_up', 'familys_up', 'paid_courses', 'activities', 'nursery',
                    'study_time_granular','thought_higher_eduaction', 'internet', 'romantic', 'family_relations',
                    'free_time', 'go_out', 'health', 'absences', 'score']

# Combine object columns (after first analysis)
obj_columns=['school','sex','address','family_size','parent_status','mother_job','father_job','reason','guardian',
             'schools_up','familys_up','paid_courses','activities','nursery','thought_higher_eduaction','internet',
             'romantic','mother_education','father_education','travel_time','study_time']

# Combine numerical columns (after first analysis)
num_columns=['age',  'failures', 'family_relations', 'free_time', 'go_out', 'health', 'absences', 'score']

# Look at the data
sns.heatmap(pupil_math.corr(), annot=True, cmap='coolwarm')
display(pupil_math.corr())
# plt.show()

# Remove rows without scores, as they have no sence
pupil_math = pupil_math.loc[pupil_math.score.notnull()]
# Remove column study_time_granular because it correlates with study_time (correlation 	coefficient is -1)
pupil_math.drop(['study_time_granular'], inplace = True, axis = 1)

# Show all unique values and it's count
for col in pupil_math.columns:
    display(f'Counts of uniqe values of column "{col}"\n',pd.DataFrame(pupil_math[col].value_counts()))

## After data review we can make next desicions:
### 1) In column "age" there aren't enough values of more than 19 for analysis. Turn them into None
### 2) In columns "mother_education" and "father_education' there are not enough values of less than 1.
#       Turn them into None
### 3) In column "father_education" there is one outlier (40). Turn it into None
### 4) In column "family_relations there is one outlier (-1). Turn it into None
### 5) Columns "absence" and "score" need more analysing for removeing outliers
### 6) Column "score" has zero (0) values. it's obviously useless data. Remove it.

# Turn to None age that does not make sence in data set.
pupil_math.age = pupil_math.age.apply(lambda x: None if x>19 else x)
# Turn to None all outliers in column father_education and mother_education
pupil_math.father_education=pupil_math.father_education.apply(lambda x: None if x<1 or x>4 else x)
pupil_math.mother_education=pupil_math.mother_education.apply(lambda x: None if x<1 or x>4 else x)
# Turn to None all outliers in column family_relations
pupil_math.family_relations=pupil_math.family_relations.apply(lambda x: None if x<1 or x>5 else x)
# Remove 0 scores from data set
pupil_math = pupil_math.loc[pupil_math.score>0]
# After analyzing data frame make the desicion to remove all rows containing values more than 34 (75 percentile)
pupil_math = pupil_math.loc[pupil_math.absences<=34]

# Desipher columns education, travel and study time for more clarity
pupil_math.father_education=pupil_math.father_education.apply\
    (lambda x: list(education.keys())[list(education.values()).index(int(x))] if not(pd.isnull(x)) else x)
pupil_math.mother_education=pupil_math.mother_education.apply\
    (lambda x: list(education.keys())[list(education.values()).index(int(x))] if not(pd.isnull(x)) else x)
pupil_math.travel_time=pupil_math.travel_time.apply\
    (lambda x: list(travel_time.keys())[list(travel_time.values()).index(int(x))] if not(pd.isnull(x)) else x)
pupil_math.study_time=pupil_math.study_time.apply\
     (lambda x: list(study_time.keys())[list(study_time.values()).index(int(x))] if not(pd.isnull(x)) else x)

# Replace all values NaN to None in all columns
for col in pupil_math.columns:
    pupil_math[col]=nan_to_none(col)

# Calculate and display outliers for column "absense"
IQR = pupil_math.absences.quantile(0.75) - pupil_math.absences.quantile(0.25) # Calculate interquartile range
perc25 = pupil_math.absences.quantile(0.25) # Calculate first quartile
perc75 = pupil_math.absences.quantile(0.75) # Calculate therd quartile
print('25-percentile: {},'.format(perc25), '75-percentile: {},'.format(perc75), "IQR: {}, ".format(IQR),
      "Range of outliers: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
display(pupil_math[pupil_math.absences>(perc75 + 1.5*IQR)][['absences','score']].describe(),
        pupil_math[pupil_math.absences>(perc75 + 1.5*IQR)][['absences','score']])

# Find significant diference for all columns
meaning_columns_d1 = []  # List for saving the information about columns that have significant diference
for col in pupil_math.columns[:-2]:  # Column absence will be test single

    obj_plots(col, pupil_math)  # Plot information about column
    # Test for significant diference
    if get_stat_dif(col,pupil_math) == 'There are significant diference for colunm ' + col:
        meaning_columns_d1.append(col)

sns.jointplot(x='absences', y='score', data=pupil_math)  # Plot the figure of depending absences and score
# plt.show()

display('Columns with significant diference', meaning_columns_d1)

sns.heatmap(pupil_math.corr(), annot=True, cmap='coolwarm')  # Display correlation heatmap for numerical columns

# First analysis allowes us to make the following conclutions
## 1) the significant columns are adress, mother_education, father_education, mother_job, study_time,
# failures, schools_up, go_out.
## 2) Column absence, age and go_out have negative correlation with score
## 3) For further analysis let us sort data set into groups.

# Analyze data groups for unsignificant columns
meaning_datas_d2=[] # List for log of data analyze
meaning_column_d2=[] # List for saving information about columns that have significant diference
columns_numb=len(pupil_math.columns.tolist()[:-2])

for i in range(columns_numb):
            for j in range(i + 1, columns_numb):

                # Exclude from analysis columns with hight significant diference
                if (pupil_math.columns[i] not in meaning_columns_d1) \
                        and (pupil_math.columns[j] not in meaning_columns_d1):

                    # create statistic table for data frames and remove values less that 5
                    data_frame = pupil_math.groupby(by=pupil_math.columns[j])['score']\
                        .agg(['count', 'mean', 'std', 'median', 'min', 'max', percentile(25), percentile(75)])
                    data_frame = data_frame[data_frame['count']>5]

                    # Check if we still have something to test
                    if len(data_frame)>1:

                        for indexes in data_frame.index:

                            # Test data frame for significant diference
                            statistic_d2 = get_stat_dif(pupil_math.columns[i],
                                                        pupil_math[pupil_math[pupil_math.columns[j]] == indexes])
                            # If data frame has significant diference write a log about it and include columns into
                            # the list
                            if statistic_d2 == 'There are significant diference for colunm ' + pupil_math.columns[i]:
                                meaning_datas_d2.append(statistic_d2 + " devided by " + pupil_math.columns[j]
                                                          + ' in ' + str(indexes))
                                display(statistic_d2 + " devided by " + pupil_math.columns[j] + ' in ' + str(indexes))
                                meaning_column_d2+=[pupil_math.columns[i], pupil_math.columns[j]]
                                # Plot the boxplot and statistic table
                                obj_plots(pupil_math.columns[i],
                                          pupil_math[pupil_math[pupil_math.columns[j]] == indexes])

# Turn list of meaning_columns_d2 into set to remove all dublicates
meaning_column_d2=set(meaning_column_d2)
# Display the log and list of columns
display(meaning_column_d2)
display(meaning_datas_d2)

# Use another method for data set. Turn all object column into numerical
pupil_math_num=pupil_math.copy()
pupil_math_num.school=pupil_math_num.school.apply(lambda x: school[x] if not(pd.isnull(x)) else x)
pupil_math_num.sex=pupil_math_num.sex.apply(lambda x: sex[x] if not(pd.isnull(x)) else x)
pupil_math_num.address=pupil_math_num.address.apply(lambda x: address[x] if not(pd.isnull(x)) else x)
pupil_math_num.family_size=pupil_math_num.family_size.apply(lambda x: family_size[x] if not(pd.isnull(x)) else x)
pupil_math_num.parent_status=pupil_math_num.parent_status.apply(lambda x: parent_status[x] if not(pd.isnull(x)) else x)
pupil_math_num.mother_education=pupil_math_num.mother_education.apply(lambda x:
                                                                      education[x] if not(pd.isnull(x)) else x)
pupil_math_num.father_education=pupil_math_num.father_education.apply(lambda x:
                                                                      education[x] if not(pd.isnull(x)) else x)
pupil_math_num.mother_job=pupil_math_num.mother_job.apply(lambda x: job[x] if not(pd.isnull(x)) else x)
pupil_math_num.father_job=pupil_math_num.father_job.apply(lambda x: job[x] if not(pd.isnull(x)) else x)
pupil_math_num.reason=pupil_math_num.reason.apply(lambda x: reason[x] if not(pd.isnull(x)) else x)
pupil_math_num.guardian=pupil_math_num.guardian.apply(lambda x: guardian[x] if not(pd.isnull(x)) else x)
pupil_math_num.travel_time=pupil_math_num.travel_time.apply(lambda x: travel_time[x] if not(pd.isnull(x)) else x)
pupil_math_num.study_time=pupil_math_num.study_time.apply(lambda x: study_time[x] if not(pd.isnull(x)) else x)
pupil_math_num.schools_up=pupil_math_num.schools_up.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.familys_up=pupil_math_num.familys_up.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.paid_courses=pupil_math_num.paid_courses.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.activities=pupil_math_num.activities.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.nursery=pupil_math_num.nursery.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.thought_higher_eduaction=pupil_math_num.thought_higher_eduaction\
    .apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.internet=pupil_math_num.internet.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)
pupil_math_num.romantic=pupil_math_num.romantic.apply(lambda x: yes_no_colm[x] if not(pd.isnull(x)) else x)

# As data set containes too many rows and columns transfer results of correlation table into Excel
pupil_math_num.corr().to_excel('./correlation_table.xlsx')
# In table we can see a hight correlation between father and mother education (0.63).
# And there is hight correlation between mother_education and mother_job (0.5)
# For removing columns we need deeper anilysis
#

# The conclusions of EDA

# Aftere EDA use the following columns for future analysis:
meaning_columns= ['school', 'sex', 'age', 'address', 'family_size', 'parent_status', 'mother_education', 'father_education',
                  'mother_job',  'father_job', 'reason', 'guardian', 'travel_time', 'study_time','failures', 'schools_up',
                  'familys_up', 'paid_courses', 'activities', 'nursery','thought_higher_eduaction', 'romantic',
                  'family_relations', 'free_time', 'go_out', 'health', 'absences', 'score']

# For further analysis use the following data set:
EDA_pupil_math=pupil_math[meaning_columns].copy()
EDA_pupil_math.info()

# We can count empty values after EDA:
EDA_pupil_math['none_value_count']=np.zeros(shape=(len(EDA_pupil_math['score'])))

for i in EDA_pupil_math.columns:
    for j in EDA_pupil_math[EDA_pupil_math[i].isnull()].index:
        EDA_pupil_math['none_value_count'][j]+=1

display(EDA_pupil_math.none_value_count.describe(),EDA_pupil_math.none_value_count.value_counts())

# Results of EDA:
## 1) Data has few empty values (441 of 9715 (4.5 %))
## 2) There are few outliers in the columns father_education, family_relations, absence
## 3) There are empty and zero values of score in data set that had been removed during EDA
## 4) In columns age, mother_education, father_education some values are present occasionally
## 5) The most significant columns are adress, mother_education, father_education, mother_job, study_time, failures,
#       schools_up, go_out
## 6) Less significant difference we find in columns activities, age, family_relations, family_size,
# familys_up, father_job, free_time, guardian, health, nursery, paid_courses, parent_status, reason, romantic, school,
# sex, thought_higher_eduaction, travel_time (see log in meaning_datas_d2)
## 7) There is hight correlation between father and mother education (0.63). To remove one of this column deeper
 # anilysis is needed
## 8) There is hight correlation between mother_education and mother_job (0.5)
## 9) In data set some columns correlate with others in some data frames (e.g. GP school has more students
# who live in town than MS school)
## 10) in general, "parents education" and "study time" have positive correlation with score
## 11) in general, age, failures, go out time, absence have negative correlation with score.