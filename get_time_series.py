import numpy as np
import os
import operator
import json
import matplotlib.pyplot as plt


class User(object):
    def __init__(self, path):
        users = {}
        for file in os.listdir(path+'/users/'):
            user = json.loads("".join(open(path+'/users/'+file).readlines()))
            users[int(file.split(".")[0])] = self.__try(user)
        self.user = users
            
    def __try(self, user):
        try:
            ers = user['followers_count']
        except:
            ers = None
        try:
            ings = user['friends_count']
        except:
            ings = None
        try:
            inter = user['statuses_count']
        except:
            inter = None
        
        return {'followers': ers, 'followings': ings, 'interactions': inter}
    
class Post(object):
    def __init__(self, path):
        self.post = {}
        for file in os.listdir(path+'/post/'):
            self.post[int(file.split(".")[0])] = json.loads("".join(open(path+'/post/'+file).readlines()))
    
class Node(object):
    def __init__(self, tripla, retweet):
        '''
        tripla: id_user, id_post, time to root
        retweet: Si el post es retweet o no
        '''
        self.user_id = tripla[0]
        self.post_id = tripla[1]
        self.timestamp = tripla[2]
        self.retweet = retweet
        
class Tree(object):
    def __init__(self, id_news, tree_str):  
        '''
        id_news: Id de la noticia
        nodes_order: Lista de posts ordenados por tiempo
        '''
        self.id = id_news
        nodes_order = []
        for line in tree_str:
            parent, child = line.strip().split("->")
            parent = self.__node_to_list(parent)
            child = self.__node_to_list(child)     
            if parent == (None, None, 0.0): 
                nodes_order.append(Node(parent, False)) # user id, tweet id, root time 
            else: 
                if parent[2] <= child[2]: # timestamp comparison
                    retweet = True if parent[1] == child[1] else False 
                    nodes_order.append(Node(child,retweet))     
        self.nodes_order = sorted(nodes_order, key = operator.attrgetter('timestamp')) # sort by timestamp asc
        self.first_time = self.nodes_order[0].timestamp
        self.last_time = self.nodes_order[-1].timestamp
        
    def __node_to_list(self, node_str):
        num = lambda s: eval(s) if not set(s).difference('0123456789. *+-/e') else None
        array_str = [num(i.strip()[1:-1]) for i in node_str[1:-1].split(",")]
        return tuple(array_str)
    
class News:
    def __init__(self, id_news, tree_str, label):
        '''
        id_news: Id de la noticia/claim
        tree: Lista de posts ordenados por tiempo del árbol de la noticia 
        label: Etiqueta de la noticia 
        '''
        self.id = int(id_news)
        self.label = label
        self.tree = Tree(self.id, tree_str)
        self.lifespan = self.tree.last_time - self.tree.first_time
        

def load_data(path='.'):
    labels = {}
    news = {}
    for label in open(path+'/label.txt').readlines():
        labels[int(label.split(":")[1][:-1])] = label.split(":")[0]   # dict: labels[news_id] = label 
    for file in os.listdir(path+'/tree/'):
        id_news = file.split(".")[0]
        news[int(id_news)] = News(id_news, open(path+'/tree/'+file).readlines(), labels[int(id_news)]) #dict:news[news_id]=News       
    return news

def load_data_users(path='.'):
    return User(path)

def load_data_posts(path='.'):
    return Post(path)

def ts_div_time_interactions(news, time, window, slide_window): # window: 5 minutes (288 samples/24 hours)
    nodes_order = news.tree.nodes_order # list of nodes      
    freq_ret, freq_rep, tiempo = ([] for i in range(3))
    if slide_window:
        step = 1
    else: 
        step = window
    for i in np.arange(news.tree.first_time, news.tree.first_time + time, step): # interval lim inf 
            f_ret = 0
            f_rep = 0
            j = 0
            while j<len(nodes_order):
                if round(i,2) <= nodes_order[j].timestamp and nodes_order[j].timestamp <= round(i+window,2):
                    if nodes_order[j].retweet:
                        f_ret += 1       
                    else:
                        f_rep += 1
                j +=1      
            freq_ret.append(f_ret)
            freq_rep.append(f_rep)
            tiempo.append(i)
    return np.array(tiempo), np.array(freq_ret), np.array(freq_rep)

def get_interactions(news, window, slide_window, time):
    '''
    news: News dictionary
    window: int. Ventana en minutos para calcular la serie de tiempo 
    slide_window: Boolean.
    time: int or False. False si se desea obtener la serie de tiempo completa. int si se desea obtener la serie 
            de tiempo separada por este valor
    '''
    label_to_idx = {'true': 0, 'non-rumor': 1, 'unverified': 2, 'false': 3}
    retweets, replies, labels = ([] for i in range(3))
    for id_ in news.keys():
        x, y_ret, y_rep = ts_div_time_interactions(news[id_], time, window, slide_window)
        labels.append(label_to_idx[news[id_].label]) 
        retweets.append(y_ret)
        replies.append(y_rep)
    
    return np.array(labels), np.array(retweets), np.array(replies)

#time news = load_data(path = '../../../Instituto fundamento de los datos/dataset_twitter/twitter16/')
#time labels, retweets, replies = get_interactions(news, 5, False, False)

def get_grad_interactions(interactions): # timeseries gradient
    grads = []
    for ts in interactions:
        grads.append(np.gradient(ts, axis=1))
        print(np.average(grads[-1]))
    return np.array(grads)

#grads = get_grad_interactions(replies)
#users = load_data_users(path = '../../../Instituto fundamento de los datos/dataset_twitter/twitter16/')
#posts = load_data_posts(path = '../../../Instituto fundamento de los datos/dataset_twitter/twitter16/')

def ts_div_time_pos_contagion(news, users, posts, time = 1000, window = 5, slide_window =  False):
    nodes_order = news.tree.nodes_order  
    x, y = ([] for i in range(2))
    users_exists = set()
    lim = 1 if slide_window else window
    j = 1
    time = time if time else news.tree.last_time
    for section in np.arange(news.tree.first_time, news.tree.last_time, time): 
        freq_cont, tiempo = ([] for i in range(2))
        for i in np.arange(section, section+time, lim): #range in tine i+window
            f_cont = 0
            while j<len(nodes_order) and nodes_order[j].timestamp<=round(i+window,2) and round(i,2)<=nodes_order[j].timestamp:
                if not nodes_order[j].user_id in users_exists:
                    try:
                        f_cont += users.user[nodes_order[j].user_id]['followers'] if users.user[nodes_order[j].user_id]['followers'] else 0
                        users_exists.add(nodes_order[j].user_id)
                    except KeyError:
                        try:
                            f_cont += users.user[int(posts.post[nodes_order[j].post_id]['user']['id_str'])]['followers']
                            users_exists.add(int(posts.post[nodes_order[j].post_id]['user']['id_str']))
                        except:
                            pass
                j += 1     
            freq_cont.append(f_cont)
            tiempo.append(i)
        y.append(np.array(freq_cont))
        x.append(np.array(tiempo))
    return np.array(x), np.array(y)

def get_pos_contagion(news, users, posts, window, slide_window, time):
    '''
    news: Diccionario de objetos News
    window: int. Ventana en minutos para calcular la serie de tiempo 
    slide_window: Boolean.
    time: int or False. False si se desea obtener la serie de tiempo completa. int si se desea obtener la serie 
            de tiempo separada por este valor
    '''
    label_to_idx = {'true': 0, 'non-rumor': 1, 'unverified': 2, 'false': 3}
    pos_cont, labels = ([] for i in range(2))
    for id_ in news.keys():
        if time:
            x, y = ts_div_time_pos_contagion(news[id_], users, posts, time = time, window = window, slide_window = slide_window)
            y = y[0]
        else:
            x, y = ts_div_time_pos_contagion(news[id_], users, posts, time = time, window = window, slide_window = slide_window)
        labels.append(label_to_idx[news[id_].label])
        pos_cont.append(y)
    
    return np.array(labels), np.array(pos_cont)

#labels, pos_cont = get_pos_contagion(news, users, posts,  5, False, False)

def get_contagion(retweets, replies, pos_cont):
    np.seterr(divide='ignore', invalid='ignore')
    return np.divide(retweets + replies, pos_cont) 

#contagion = get_contagion(retweets, replies, pos_cont)