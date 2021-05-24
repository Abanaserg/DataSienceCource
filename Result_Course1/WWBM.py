import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import display

# счетчик уникальных значений в массиве при учете того, что разделитель "|"
def un_name_count (df):
    names={} # создание словаря из имен (ключ) и подсчета количества уникальных значений
    for i in df.index:
        names_str=list(df[i].split('|'))
        for j in names_str:
            if j in names.keys():
                names[j]+=1 # если имя уже есть в словаре, то увеличиваем счетчик
            else:
                names[j]=1 # если имени нет в словаре, начинаем счетчик и создаем ключ
    return names # возвращаем словарь по результаам функции

# подсчет прибыльных и убыточных жанров
# names - перечень уникальных значений, names_list - список состоящий из наборов names, profit -
# массив разницы между сборами и бюджетом
def profit_count(names,names_list,profit):
    names_dict=dict.fromkeys(names)

    for i in names_dict.keys():
        names_dict[i] = [0 ,0]

    for i in names_list.index:
        for j in names:
            if j in names_list[i]:
                # если прибыль больше 0 увеличивает счетчик "прибыльности". если меньше 0, счетчик убыточности
                if profit.iloc[i]>0:
                    names_dict[j][0]+=1
                else:
                    names_dict[j][1] += 1

    return names_dict

# функия для суммирования значений
# names - перечень уникальных значений, names_list - список состоящий из наборов names,
# values - массив значений по которым выполняется суммирование
def summ_values(names,names_list,values):
    names_dict = dict.fromkeys(names)

    for i in names_dict.keys():
        names_dict[i] = 0

    for i in names_list.index:
        for j in names:
            if j in names_list[i]:
                # проверка на тип. если строка, считать количество. В остальных случаях выполнять суммирование.
                # Подразумевается, что входными данными могут быть int, float или str
                if type(values[i]) is not str:
                    names_dict[j]+=values[i]
                else:
                    names_dict[j]+=1

    return names_dict


#функция для подсчета вхождения значений списка names в серии names_list и значения s_name в серии s_name_list
# подразумевается, что размерности s_name_list и names_list одинаковые и формирующиеся из одногои тогоже DF
def count_smt(names,s_name,names_list,s_name_list):
    names_dict = dict.fromkeys(names)

    for i in names_dict.keys():
        names_dict[i] = 0

    for i in names_list.index:
        for j in names:
            if (j in names_list[i]) and (s_name in s_name_list[i]):
                names_dict[j]+=1
    return names_dict

# функция приведения дат к единому формату dd.mm.yyyy
def date_standart(df):
    for i in df.index:
        if '/' in df[i]:
            date=list(df[i].split('/'))
            if len(date[0])==1: #перевод месяца к формату mm
                date[0]='0'+date[0]
            df[i]=date[1]+'.'+date[0]+'.'+date[2]
    return df

#функция для подсчета длины строки и количества слов
def mid_len(names,names_list,str_list,meth='str'):
    names_dict = dict.fromkeys(names)
    for i in names_dict.keys():
        names_dict[i] = 0

    if meth=='str': # в случае если нужно посчитать количество элементов в строке
        for i in names_list.index:
            for j in names:
                # print(names_list[i])
                # print(names_list.iloc[i])
                if (j in list(names_list[i].split('|'))):
                    names_dict[j]+=len(str_list[i].replace(' ',''))

    elif meth=='word': # в случае если нужно посчитать количество слов в строке
        for i in names_list.index:
            for j in names:
                if (j in list(names_list[i].split('|'))):
                    names_dict[j]+=len(list(str_list[i].split()))


    return names_dict

# функция для подсчета частоты вхождения элементов names в names_list
def friends(names,names_list):
    l_names = len(names) # что бы несколько раз не высичлять, маленькая "оптимизация" )

    # создания DF размеров l_names х l_names изначально заполненой нулями
    names_df = pd.DataFrame([[0]*l_names]*l_names
                            ,columns=names
                            ,index=names)

    for i in names_list:
        tmp_list=list(i.split('|')) #выделение отдельных элементов из строки

        # увеличение каждого элемента массива на 1 как индикатора, что элементы tmp_list
        # входят в общий DF на конкретные позиции. при этом исключается элемент, замкнутый на себя
        for j in tmp_list:
            for k in tmp_list:
                if j != k:
                    names_df.loc[j][k]+=1
    return names_df

data = pd.read_csv('D:\Studing\SFDS_Course\Result_Course1\movie_bd_v5.csv')

data['release_date']=date_standart(data['release_date']) # перевод всех дат в тип dd.mm.yyyy
data['month']=data['release_date'].str.split('.').str[-2] # создание колонки месяца выпуска
data['title_len']=data['original_title'].str.replace(' ','').str.len() # создание колонки с длиной названия фильма
data['profit']=data.revenue-data.budget # создание колонки "прибыль"

answers=dict.fromkeys('Ответ на вопрос '+str(i) for i in range(1,28))
i=1

#Вопрос 1. У какого фильма из списка самый большой бюджет?
answers['Ответ на вопрос '+str(i)]=data[data.budget==data.budget.max()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1


#Вопрос 2. Какой из фильмов самый длительный (в минутах)?
answers['Ответ на вопрос '+str(i)]=data[data.runtime==data.runtime.max()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1

# #Вопрос 3. Какой из фильмов самый короткий (в минутах)?
answers['Ответ на вопрос '+str(i)]=data[data.runtime==data.runtime.min()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1

# #Вопрос 4. Какова средняя длительность фильмов?
answers['Ответ на вопрос '+str(i)]=round(data.runtime.mean())
print(f'ответ на Вопрос № {i} найден')
i+=1

# # Вопрос 5. Каково медианное значение длительности фильмов?
answers['Ответ на вопрос '+str(i)]=round(data.runtime.median())
print(f'ответ на Вопрос № {i} найден')
i+=1

# # Вопрос 6. Какой фильм самый прибыльный?
# # Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма.
# # Прибыль = сборы - бюджет (profit = revenue - budget).
answers['Ответ на вопрос '+str(i)]=data[data.profit==data.profit.max()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1


# # Вопрос 7. Какой фильм самый убыточный?
answers['Ответ на вопрос '+str(i)]=data[data.profit==data.profit.min()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1


# # Вопрос 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?
answers['Ответ на вопрос '+str(i)]=data[data.revenue>data.budget].original_title.count()
print(f'ответ на Вопрос № {i} найден')
i+=1



# # Вопрос 9. Какой фильм оказался самым кассовым в 2008 году?
# можно сделать и через доп. массив, как в вопросе 10
answers['Ответ на вопрос '+str(i)]=data.iloc[data[data.release_year==2008].revenue.sort_values(ascending=False).head(1)
    .index].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1

# # Вопрос 10. Самый убыточный фильм за период с 2012 по 2014 годы (включительно)?
tmp_df=data[(data.release_year>=2012)&(data.release_year<=2014)] # промежуточный массив данных для дальнейшей обработки
answers['Ответ на вопрос '+str(i)]=tmp_df[tmp_df.profit==tmp_df.profit.min()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1


#
# # Вопрос 11. Какого жанра фильмов больше всего?

# создание отдельного DF для жанров
genres_df=pd.DataFrame.from_dict(un_name_count(data.genres), orient='index')
genres_df.columns=['counts']

answers['Ответ на вопрос '+str(i)]=genres_df[genres_df.counts==genres_df.counts.max()].counts
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# Вопрос 12. Фильмы какого жанра чаще всего становятся прибыльными?
genres_df[['profit','not_profit']]=pd.DataFrame.from_dict(profit_count(genres_df.index,data.genres,data.profit)
                                                          ,orient='index')
answers['Ответ на вопрос '+str(i)]=genres_df[genres_df.profit==genres_df.profit.max()].profit
print(f'ответ на Вопрос № {i} найден')
i+=1


# # Вопрос 13. У какого режиссёра самые большие суммарные кассовые сборы?

#Создание DF для режисеров
director_df=pd.DataFrame.from_dict(un_name_count(data.director), orient='index')
director_df.columns=['counts']
director_df['summ_revenue']=pd.DataFrame.from_dict(summ_values(director_df.index,data.director,data.revenue)
                                                   , orient='index')
answers['Ответ на вопрос '+str(i)]=director_df[director_df.summ_revenue==director_df.summ_revenue.max()].summ_revenue
print(f'ответ на Вопрос № {i} найден')
i+=1
#
#
# Вопрос 14. Какой режиссер снял больше всего фильмов в стиле Action?
director_df['Act_count']=pd.DataFrame.from_dict(count_smt(director_df.index,'Action',data.director,data.genres)
                                                , orient='index')
answers['Ответ на вопрос '+str(i)]=director_df[director_df.Act_count==director_df.Act_count.max()].Act_count
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# # Вопрос 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году?

# создание DF для актеров
actors_df=pd.DataFrame.from_dict(un_name_count(data.cast), orient='index')
actors_df.columns=['counts']
actors_df['summ_revenue_2012']=pd.DataFrame.from_dict(summ_values(actors_df.index,data[data.release_year==2012].cast
                                      ,data[data.release_year==2012].revenue), orient='index')
answers['Ответ на вопрос '+str(i)]=actors_df[actors_df.summ_revenue_2012==actors_df.summ_revenue_2012.max()]\
    .summ_revenue_2012
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# # Вопрос 16. Какой актер снялся в большем количестве высокобюджетных фильмов?
# # Примечание: в фильмах, где бюджет выше среднего по данной выборке.
actors_df['hight_budget_count']=pd.DataFrame.from_dict(summ_values(actors_df.index,data[data.budget>=data.budget.mean()].cast
                                      ,data[data.budget>=data.budget.mean()].original_title), orient='index')
answers['Ответ на вопрос '+str(i)]=actors_df[actors_df.hight_budget_count==actors_df.hight_budget_count.max()].hight_budget_count
print(f'ответ на Вопрос № {i} найден')
i+=1



# # Вопрос 17. В фильмах какого жанра больше всего снимался Nicolas Cage?
genres_df['NC']=pd.DataFrame.from_dict(count_smt(genres_df.index,'Nicolas Cage',data.genres,data.cast)
                                                , orient='index')
answers['Ответ на вопрос '+str(i)]=genres_df[genres_df.NC==genres_df.NC.max()].NC
print(f'ответ на Вопрос № {i} найден')
i+=1



# # Вопрос 18. Самый убыточный фильм от Paramount Pictures?
pmp_df=data[data.production_companies.str.contains('Paramount Pictures')]
answers['Ответ на вопрос '+str(i)]=pmp_df[pmp_df.profit==pmp_df.profit.min()].original_title
print(f'ответ на Вопрос № {i} найден')
i+=1

#
#
# # Вопрос 19. Какой год стал самым успешным по суммарным кассовым сборам?
best_year=data.groupby('release_year')['revenue'].sum()
answers['Ответ на вопрос '+str(i)]=best_year[best_year==best_year.max()]
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# # Вопрос 20. Какой самый прибыльный год для студии Warner Bros?
best_year_wb=data[data.production_companies.str.contains('Warner Bros')].groupby('release_year')['profit'].sum()
answers['Ответ на вопрос '+str(i)]=best_year_wb[best_year_wb==best_year_wb.max()]
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# # Вопрос 21. В каком месяце за все годы суммарно вышло больше всего фильмов?
best_month=data.groupby('month')['imdb_id'].count()
answers['Ответ на вопрос '+str(i)]=best_month[best_month==best_month.max()]
print(f'ответ на Вопрос № {i} найден')
i+=1


#
# # Вопрос 22. Сколько суммарно вышло фильмов летом (за июнь, июль, август)?

answers['Ответ на вопрос '+str(i)]=best_month[['06','07','08']].sum()
print(f'ответ на Вопрос № {i} найден')
i+=1

#

# # Вопрос 23. Для какого режиссера зима — самое продуктивное время года?
director_df['films_count']=pd.DataFrame.from_dict(summ_values(director_df.index,data[(data.month=='12')
                                                                                     |(data.month=='01')
                                                                                     |(data.month=='02')]
                                        .director,data[(data.month=='12')|(data.month=='01')|(data.month=='02')]
                                        .original_title), orient='index')
answers['Ответ на вопрос '+str(i)]=director_df[director_df.films_count==director_df.films_count.max()].films_count
print(f'ответ на Вопрос № {i} найден')
i+=1
#
# # Вопрос 24. Какая студия даёт самые длинные названия своим фильмам по количеству символов?
# рекомендуется добавить примечание, что бы понимать включать ли в поиск символ пробела или нет.
# Кроме того не до конца понятно само задание. имеется ввиду среднее или в принципе самое длинное

prod_comp_df=pd.DataFrame.from_dict(un_name_count(data.production_companies), orient='index')
prod_comp_df.columns=['counts']
tmp_df=pd.DataFrame.from_dict(mid_len(prod_comp_df.index,data.production_companies,data.original_title), orient='index')
prod_comp_df['mid_title_len']=[tmp_df.iloc[i][0]/prod_comp_df.iloc[i].counts for i in range(len(tmp_df))]
answers['Ответ на вопрос '+str(i)]=prod_comp_df[(prod_comp_df.mid_title_len==prod_comp_df.mid_title_len.max())]\
    .mid_title_len
print(f'ответ на Вопрос № {i} найден')
i+=1

#
# # Вопрос 25. Описания фильмов какой студии в среднем самые длинные по количеству слов?
tmp_df=pd.DataFrame.from_dict(mid_len(prod_comp_df.index,data.production_companies,data.overview,'word'), orient='index')
prod_comp_df['mid_overview_words']=[tmp_df.iloc[i][0]/prod_comp_df.iloc[i].counts for i in range(len(tmp_df))]
answers['Ответ на вопрос '+str(i)]=prod_comp_df[(prod_comp_df.mid_overview_words==prod_comp_df.mid_overview_words.max())]\
    .mid_overview_words
print(f'ответ на Вопрос № {i} найден')
i+=1



# # Вопрос 26. Какие фильмы входят в один процент лучших по рейтингу?
answers['Ответ на вопрос '+str(i)]=data[['original_title','vote_average']].sort_values(['vote_average']
                                                                            ,ascending=False,).head(int(len(data)*0.01))
print(f'ответ на Вопрос № {i} найден')
i+=1


#
# # Вопрос 27. Какие актеры чаще всего снимаются в одном фильме вместе?
tmp_df=friends(actors_df.index,data.cast)
tmp_df['max_counts']=tmp_df.max(axis='index')
answers['Ответ на вопрос '+str(i)]=tmp_df[tmp_df.max_counts==tmp_df.max_counts.max()].index
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(answers)
