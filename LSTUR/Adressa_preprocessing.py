# %%
import os
import json
import time
from datetime import datetime
from random import sample
# import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import itertools


############################# 최종 전처리 코드

NEWS_DIRECTORY = './1w_light/one_week'
dir_list = ['history','train','test']

def parse_time(timestamp):
    return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

def str_to_timestamp(string):
    return datetime.timestamp(datetime.strptime(string,'%Y-%m-%d %H:%M:%S'))

# def make_user_news_data():
print()
print('Make_user_news_data',end='    ')
print('all_start_time:',time.strftime('%y-%m-%d %H:%M:%S'))
print()
print()
start = time.time()
first_execution = True
def make_user_news_dic(body):
    global user_dic
    global user_cnt
    global news_cnt
    global news_dic
    global index
    global behaviors_list
    global filter1_cnt
    global behaviors_user_filter_cnt
    global time_error_cnt
    global first_execution
    

    """유효한 뉴스 데이터만 취급
    이로써, 뉴스 데이터는 시작부터 필터링 됨"""
    filter_list = ["time", "publishtime",
                "title", "userId","url", "canonicalUrl"]
    if all(filter in body.keys() for filter in filter_list):
        """내가 만든 부분으로, 처음 실행 때는 user_filter가 정의되지 않아
        이를 제하고 실행하게 함"""
        
        if first_execution:
            pass
        else:
            if user_filter_dic.get(body['userId'],False):
                behaviors_user_filter_cnt += 1
                return
        
        """내 주석: news_dic 만들기"""
        body['publishtime'] = body['publishtime'].split('.')[0].replace('T',' ')
        body['time'] = parse_time(body['time'])
        
        if str_to_timestamp(body['time'])-str_to_timestamp(body['publishtime']) < 0:
            time_error_cnt += 1
            return
        
        if body['canonicalUrl'] not in news_dic.keys():
            news_id = 'N'+str(news_cnt)
            news_dic[body['canonicalUrl']] = [news_id,body['publishtime'],body['title']]
            news_dic[body['canonicalUrl']].append([body['time']])
            news_cnt += 1
            # news_df = news_df.append({'news_id':news_id,'url':body['canonicalUrl'],'publish_time':body['publishtime'],'clicked_times':body['time'],'title':body['title']},ignore_index=True)
            if 'category1' in body.keys():
                news_dic[body['canonicalUrl']].append(body['category1'])
            else:
                news_dic[body['canonicalUrl']].append("")
            # if news_cnt == 10:
            #     break
        else:
            news_id = news_dic[body['canonicalUrl']][0]
            news_dic[body['canonicalUrl']][3].append(body['time'])
            # target_news_index = int(news_dic[body['canonicalUrl']][1:]) - 1
            # news_id = news_df.loc[target_news_index]['news_id']
            # total_click = news_df.loc[target_news_index]['clicked_times']+',' + body['time']
            # news_df.at[target_news_index,'clicked_times'] = total_click
        """ 내 주석: 
        news_dic 만들기 끝
        user_dic 만들기 """
        add_click = news_id+ ','+body['publishtime']+ ','+body['time']
        if body['userId'] not in user_dic.keys():
            user_id = 'U'+str(user_cnt)
            user_dic[body['userId']] =[user_id]
            user_dic[body['userId']].append([])
            user_dic[body['userId']].append([])
            user_dic[body['userId']].append([])
            user_dic[body['userId']][index].append([add_click])
            user_cnt += 1

        user_dic[body['userId']][index].append([add_click])

        if data_type == "train":
            user_id = user_dic[body['userId']][0]
            history_list = list(set([j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]))
            write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
            behaviors_list.append(write_line)

            # target_user_index = int(user_dic[body['userId']][1:])-1
            # total_click = user_df.loc[target_user_index][data_type]+'\t'+add_click
            # user_df.at[target_user_index,data_type] = total_click
        elif data_type == "test":
            user_id = user_dic[body['userId']][0]
            """ 내 주석: history_list에는 history와 train 모두 사용 """
            history_list = list(set(
                                    [j.split(',')[0] for i in user_dic[body['userId']][1] for j in i ]
                                    + [j.split(',')[0] for i in user_dic[body['userId']][2] for j in i ]))
            write_line = user_id+'\t'+body['time'] + '\t' + ' '.join(history_list) + '\t'+news_id+'-1'
            behaviors_list.append(write_line)

    else:
        filter1_cnt += 1




"""make_user_news_dic 정의 끝!!!
이제는 make_user_news_dic 실행하여 user.tsv 생성"""

########### user_behaviors filtering & news.tsv, user_behaviors.tsv
index = 0

user_cnt = 1
news_cnt = 1
user_dic = {}
behavior_cnt = 0
total_cnt = 1

filter1_cnt = 1  ### 더미 데이터 제거
behaviors_user_filter_cnt = 1
time_error_cnt = 1

# start preprocessing:
print()
print('Start preprocessing',end='    ')
print('preprocessing start_time:',time.strftime('%y-%m-%d %H:%M:%S'))
print()
print()

A = 1000000
for data_type in dir_list:
    # news_df = pd.DataFrame(columns = ['news_id','publish_time','category','clicked_times','title','url'])
    news_dic = {}
    behaviors_list= []
    length = len(dir_list)
    index += 1
    print(f'[{data_type}] {index} / {length}',end='     ')
        # print('[missed data_type]', cnt)
    data_folder = os.path.join(NEWS_DIRECTORY, data_type)
    print(f'위치 :  {data_folder}')
    print()
    index2 = 1
    for news in os.listdir(data_folder):
        start2 = time.time()
        if news.endswith('.tsv'):
            continue
        news_file = os.path.join(data_folder,news)
        print(f'{index2}/{len(os.listdir(data_folder))}     파일:{news_file}',end=' #')
        print('start_time:',time.strftime('%m-%d %H:%M:%S'),end=' #')


        index2 += 1
        file1 = open(news_file,'r')
        while True:
            total_cnt += 1
            line = file1.readline()
            if not line:
                break
            body = json.loads(line)
            # print(body.keys())

            make_user_news_dic(body)

        print('elapsed_time:',round(time.time()-start2,1))
        a = len(user_dic)
        # if a >= A:
        #     print(f"A: {A}, a: {a} \n")
        #     exit()
            
        print(f"A: {A}, a: {a} \n")
        A = a

print()
print(len(user_dic))
print()
print()        
for key, value in itertools.islice(user_dic.items(), 10):
    print(f"{key}")
print()
print()
# print("exit")
# exit()
 
        
########## user filtering & user.tsv


user_filter = []   ### 각 구간 별 클릭 없는 유저 제거
user_filter2_cnt = 1
user_filter3_cnt = 1
user_time_list = []
with open(os.path.join(NEWS_DIRECTORY,'user.tsv'),'w',encoding='UTF-8') as wf2:
    for key in tqdm(user_dic.keys()):
        if len(user_dic[key][1]) + len(user_dic[key][2]) + len(user_dic[key][3]) < 20:
            user_filter2_cnt += 1
            user_filter.append(key)

        elif (len(user_dic[key][1]) == 0) or (len(user_dic[key][2]) == 0) or (len(user_dic[key][3]) == 0):
            user_filter3_cnt += 1
            user_filter.append(key)
        else:
            writeline = user_dic[key][0]+'\t'
            for click in user_dic[key][1]:
                click = click[0]
                writeline += click+';'
                tmp = click.split(',')
                user_time_list.append(str_to_timestamp(tmp[2])-str_to_timestamp(tmp[1]))
            writeline.strip(';')
            writeline += '\t'
            for click in user_dic[key][2]:
                click = click[0]
                writeline += click+';'
                tmp = click.split(',')
                user_time_list.append(str_to_timestamp(tmp[2])-str_to_timestamp(tmp[1]))
            writeline.strip(';')
            writeline += '\t'
            for click in user_dic[key][3]:
                click = click[0]
                writeline += click+';'
                tmp = click.split(',')
                user_time_list.append(str_to_timestamp(tmp[2])-str_to_timestamp(tmp[1]))
            writeline.strip(';')
            writeline += '\n'
            wf2.write(writeline)
#     news_df.to_csv(os.path.join(data_folder,'news.tsv'),index=False)
# user_df.to_csv(os.path.join(NEWS_DIRECTORY,'user.tsv'),index=False)
print()
print(user_filter2_cnt)
print(user_filter3_cnt)
print()
print()
print('total_elapsed_time:',round(time.time()-start,1))
user_filter = set(user_filter)
user_filter_dic = {x:True for x in user_filter}  
        
        
        
        
        
        
        
""" make_user_news_dic 실행 끝.
이제는 얻어진 user_filter_dic을 사용하여 make_user_news_dic 다시 생성"""      


########### user_behaviors filtering & news.tsv, user_behaviors.tsv
index = 0

user_cnt = 1
news_cnt = 1
user_dic = {}
behavior_cnt = 0
total_cnt = 1

filter1_cnt = 1  ### 더미 데이터 제거
behaviors_user_filter_cnt = 1
time_error_cnt = 1

# start preprocessing:
print()
print('Start preprocessing',end='    ')
print('preprocessing start_time:',time.strftime('%y-%m-%d %H:%M:%S'))
print()
print()

for data_type in dir_list:
    # news_df = pd.DataFrame(columns = ['news_id','publish_time','category','clicked_times','title','url'])
    news_dic = {}
    behaviors_list= []
    length = len(dir_list)
    index += 1
    print(f'[{data_type}] {index} / {length}',end='     ')
        # print('[missed data_type]', cnt)
    data_folder = os.path.join(NEWS_DIRECTORY, data_type)
    print(f'위치 :  {data_folder}')
    print()
    index2 = 1
    for news in os.listdir(data_folder):
        start2 = time.time()
        if news.endswith('.tsv'):
            continue
        news_file = os.path.join(data_folder,news)
        print(f'{index2}/{len(os.listdir(data_folder))}     파일:{news_file}',end=' #')
        print('start_time:',time.strftime('%m-%d %H:%M:%S'),end=' #')


        index2 += 1
        file1 = open(news_file,'r')
        while True:
            total_cnt += 1
            line = file1.readline()
            if not line:
                break
            body = json.loads(line)
            # print(body.keys())
            
            """여기서 first_execution=False이므로, user filtering 재수행"""
            first_execution = False
            make_user_news_dic(body)

        print('elapsed_time:',round(time.time()-start2,1))


    """수행된 유저 필터링을 바탕으로 최종 news/behaviors.tsv 생성"""
    with open(os.path.join(data_folder,'news.tsv'),'w',encoding='UTF-8') as wf:
        for key in news_dic.keys():
            wf.write(news_dic[key][0]+'\t'+news_dic[key][1]+'\t'+news_dic[key][2]+'\t'+','.join(news_dic[key][3])+'\t'+news_dic[key][4]+'\n')
    with open(os.path.join(data_folder,'behaviors.tsv'),'w',encoding='UTF-8') as wf:
        for line in behaviors_list:
            wf.write(line+'\n')
    behavior_cnt += len(behaviors_list)
    print(f"{data_type} behavior_cnt",len(behaviors_list))





print("news_cnt:",news_cnt)
print("user_cnt",user_cnt)
print("behavior_cnt",behavior_cnt)
print("total_cnt",total_cnt)
print("filter1_cnt",filter1_cnt)