import numpy as np
import pandas as pd
import re
def load_data():
    users_title=['UserID','Gender','Age','JobID','Zip-code']
    users=pd.read_table('./users.dat',sep='::',header=None,names=users_title,engine='python')
    users=users.filter(regex='UserID|Gender|Age|JobID')
    users_orig=users.values
    gender_map={'F':0,'M':1}
    users['Gender']=users['Gender'].map(gender_map)
    age_map={val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age']=users['Age'].map(age_map)
    # print(set(users['Age']))
    # print(users)
    movies_title=['MovieID','Title','Genres']
    movies=pd.read_table('./movies.dat',sep='::',header=None,names=movies_title,engine='python')
    movies_orig=movies.values
    pattern=re.compile(r'^(.*)\((\d+)\)$')
    title_map={val:pattern.match(val).group(1)for ii,val in enumerate(set(movies['Title']))}
    movies['Title']=movies["Title"].map(title_map)
    genres_set=set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('<PAD>')
    genres2int={val:ii for ii ,val in enumerate(genres_set)}
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

    movies['Genres'] = movies['Genres'].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    movies['Title'] = movies['Title'].map(title_map)

    # print(genres_set)

    ratings_title=['UserID','MovieID','ratings','timestamps']
    ratings=pd.read_table('./ratings.dat',sep='::',header=None,names=ratings_title,engine='python')
    ratings=ratings.filter(regex='UserID|MovieID|ratings')
    data=pd.merge(pd.merge(ratings,users),movies)
    target_fields=['ratings']
    features_pd,targets_pd=data.drop(target_fields,axis=1),data[target_fields]
    features=features_pd.values
    targets_values=targets_pd.values
    return title_count,title_set,genres2int,features,targets_values,ratings,users,movies,data,movies_orig,users_orig
load_data()