import pickle
import pandas as pd
import torch
from tqdm import tqdm
import os

"""
마찬가지로 category2int 무쓸모
"""
def make_test_datas(snapshots_num: int):
    # a) test dataset(0212 08:00:02 ~ 0219 08:00:01)인 valid_tkg_behaviors.tsv 로드
    test_file_path = 'psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
    test_df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
    # click_time을 string에서 datetime으로 변환
    test_df['click_time'] = pd.to_datetime(test_df['click_time'])
    # ### half 버전!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # test_time = pd.Timestamp('2017-02-15 20:00:01')
    # test_df  = test_df[test_df['click_time'] <= test_time]
    
    # test_df에서 nan이 존재하는 행 제거
    test_df = test_df.dropna(subset=['clicked_news'])

    ########################################### 여기부터 negative sampling
    # news2int 가져오기
    news2int_file_path = 'psj/Adressa_4w/history/news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')
    # news2int를 dictionary로 변환
    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
    # user2int mapping
    user2int_df = pd.read_csv(os.path.join('psj/Adressa_4w/history/', 'user2int.tsv'), sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    all_user_ids = user2int_df['user_int'].tolist()   # 0 ~ 84988

    test_df['user_int'] = test_df['history_user'].map(user2int)
    test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)


    """
    test_news: 각 요소(리스트)는 test data에서 각 유저가 클릭한 news_ids
    - shape: (user_num, test data에서 각 유저의 클릭 수)

    test_time: 각 요소(리스트)는 test data에서 각 유저가 클릭한 뉴스의 times
    - shape: (user_num, test data에서 각 유저의 클릭 수)
    """
    test_news = []
    test_time = []
    test_empty_check = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(test_df[test_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        test_news.append(u_news)
        
        u_len = len(test_df[test_df['user_int'] == u_id])
        u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
        test_time.append(u_time)
        if u_len == 0:
            test_empty_check.append(False)
        else:
            test_empty_check.append(True)
        
    
    return list(zip(test_news, test_time, test_empty_check))







# import pickle
# import pandas as pd
# import torch
# from tqdm import tqdm
# import datetime
# import os

# """
# train_news, train_category는 어차피 안 쓰임
# model의 forward input이지만, 안 쓰임
# -> category2int 바꾸든 안 바꾸든 현재는 의미 없음
# """

# def make_train_datas(week = 7):

#     # snapshots에 카테고리 정보 추가하기
#     # 1. history, train, test에 대해 전역 category2int를 만든다 (이미 있음)
#     # 2. category2int를 g.edges['clicked'].data['category']에 저장한다 (이미 있음)
#     # 3. main.py에서 데이터 로드할 때 category idx를 사용한다
#     # *** 전역 news2int도 필요!!!


#     # news2int 가져오기
#     news2int_file_path = f'psj/Adressa_{week}w/history/news2int.tsv'
#     news2int = pd.read_csv(news2int_file_path, sep='\t')

#     # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
#     # train_file_path = f'psj/Adressa_{week}w/datas/history_tkg_behaviors.tsv'
#     # df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
#     # # click_time을 string에서 datetime으로 변환
#     # df['click_time'] = pd.to_datetime(df['click_time'])
    
#     # criteria_time1 = pd.Timestamp('2017-01-20 00:00:00')
#     # criteria_time2 = pd.Timestamp('2017-01-23 00:00:00')
#     # train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
#     train_df = pd.read_csv(f'psj/Adressa_{week}w/train/valid_tkg_behaviors.tsv', sep='\t')
    
#     # train_df에서 nan이 존재하는 행 제거
#     train_df = train_df.dropna(subset=['clicked_news'])



#     ########################################### 여기부터 negative sampling을 위해 추가된 부분
#     # news2int를 dictionary로 변환
#     news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
#     # user2int mapping
#     user2int_df = pd.read_csv(os.path.join(f'psj/Adressa_{week}w/datas/', 'user2int.tsv'), sep='\t')
#     user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
#     all_user_ids = user2int_df['user_int'].tolist()   # 0 ~ 84988

#     # train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
#     train_df['user_int'] = train_df['history_user'].map(user2int)
#     train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)
#     category2int = pd.read_csv('psj/Adressa_4w/datas/category2int_nyheter_splitted.tsv', sep='\t')
#     # 필요시 category2int에 'No category' 추가
#     if 'No category' not in category2int['category'].values:
#         new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
#         category2int = pd.concat([new_row, category2int], ignore_index=True)
#     cat2int = category2int.set_index('category')['int'].to_dict()
    
#     ############# category가 nyheter이면 subcategory로 매핑, 그렇지 않으면 category로 매핑
#     def get_cat_int(row):
#         if row['category'] == 'nyheter':
#             # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
#             return cat2int.get(row['subcategory'], cat2int['No category'])
#         else:
#             return cat2int.get(row['category'], cat2int['No category'])

#     train_df['cat_int'] = train_df.apply(get_cat_int, axis=1)
    

#     # period_start -> time_idx 매핑(0부터 시작)
#     def get_period_start(click_time, interval_minutes=1440, start_time=datetime.time(8, 0, 2)):

#         base_start = datetime.datetime.combine(click_time.date(), start_time)
#         if click_time < base_start:
#             base_start -= datetime.timedelta(days=1)
#         delta = click_time - base_start
#         periods = int(delta.total_seconds() // (interval_minutes * 60))

#         return base_start + datetime.timedelta(minutes=interval_minutes * periods)

#     train_df['click_time'] = pd.to_datetime(train_df['click_time'])
#     train_df['Period_Start'] = train_df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=30))
    
#     ### 매우 중요!!!
#     history_weeks = 5
#     snapshots_num = int(history_weeks * 7 * 24 * 2)   # 1680
#     unique_period_starts = train_df['Period_Start'].unique()
#     time_dict = {ps: i+snapshots_num for i, ps in enumerate(sorted(unique_period_starts))}
#     train_df['time_idx'] = train_df['Period_Start'].map(time_dict)

#     """
#     train_news: 각 요소(리스트)는 train data에서 각 유저가 클릭한 news_ids
#     - shape: (user_num, train data에서 각 유저의 클릭 수)

#     train_category: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 categories
#     - shape: (user_num, train data에서 각 유저의 클릭 수)

#     train_time: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 times (snapshot or timestamp indicies)
#     - shape: (user_num, train data에서 각 유저의 클릭 수)
#     """
#     train_news = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_news = torch.tensor(train_df[train_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
#         train_news.append(u_news)
        
#     train_category = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_category = torch.tensor(train_df[train_df['user_int'] == u_id]['cat_int'].values, dtype=torch.long)
#         train_category.append(u_category)

#     train_time = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_time = torch.tensor(train_df[train_df['user_int'] == u_id]['time_idx'].values, dtype=torch.long)
#         train_time.append(u_time)
        
#     # print(train_time[0])
#     # print(len(train_time[0]))


#     # with open('./psj/Adressa_4w/train/train_datas.pkl', 'wb') as f:
#     #     pickle.dump(list(zip(train_news, train_category, train_time)), f)
    
#     return list(zip(train_news, train_category, train_time))


# # import pickle
# # import pandas as pd
# # import torch
# # from tqdm import tqdm
# # import datetime
# # import os

# # """
# # full_news_encoder.py를 사용하기 위해 category2int_pio.tsv로 변경
# # """

# # def make_train_datas():

# #     # snapshots에 카테고리 정보 추가하기
# #     # 1. history, train, test에 대해 전역 category2int를 만든다 (이미 있음)
# #     # 2. category2int를 g.edges['clicked'].data['category']에 저장한다 (이미 있음)
# #     # 3. main.py에서 데이터 로드할 때 category idx를 사용한다
# #     # *** 전역 news2int도 필요!!!


# #     # news2int 가져오기
# #     news2int_file_path = '/home/user/pyo/psj/Adressa_4w/history/news2int.tsv'
# #     news2int = pd.read_csv(news2int_file_path, sep='\t')

# #     # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
# #     train_file_path = '/home/user/pyo/psj/Adressa_4w/train/valid_tkg_behaviors.tsv'
# #     train_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
# #     # 'clicked_news' 열에서 '-1' 제거
# #     train_df['clicked_news'] = train_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

# #     # train_df에서 nan이 존재하는 행 제거
# #     train_df = train_df.dropna(subset=['clicked_news'])



# #     ########################################### 여기부터 negative sampling을 위해 추가된 부분
# #     # news2int를 dictionary로 변환
# #     news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
# #     # user2int mapping
# #     file_path = '/home/user/pyo/psj/Adressa_4w/history/history_tkg_behaviors.tsv'
# #     history_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
# #     history_df['click_time'] = pd.to_datetime(history_df['click_time'])
# #     end_time = pd.Timestamp('2017-02-05 08:00:01')
# #     history_df = history_df[history_df['click_time'] <= end_time]   # 정확히 5주 데이터만 사용하도록 필터링
# #     users = history_df['history_user'].unique()
# #     user2int_df = pd.read_csv(os.path.join('/home/user/pyo/psj/Adressa_4w/history/', 'user2int.tsv'), sep='\t')
# #     user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
# #     all_user_ids = [i for i in range(len(users))]   # 0 ~ 84988

# #     # train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
# #     train_df['user_int'] = train_df['history_user'].map(user2int)
# #     train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)
# #     category2int = pd.read_csv('/home/user/pyo/category2int_nyheter_splitted.tsv', sep='\t')
# #     # 필요시 category2int에 'No category' 추가
# #     if 'No category' not in category2int['category'].values:
# #         new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
# #         category2int = pd.concat([new_row, category2int], ignore_index=True)
# #     cat2int = category2int.set_index('category')['int'].to_dict()
    
# #     ############# category가 nyheter이면 subcategory로 매핑, 그렇지 않으면 category로 매핑
# #     def get_cat_int(row):
# #         if row['category'] == 'nyheter':
# #             # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
# #             return cat2int.get(row['subcategory'], cat2int['No category'])
# #         else:
# #             return cat2int.get(row['category'], cat2int['No category'])

# #     train_df['cat_int'] = train_df.apply(get_cat_int, axis=1)
    

# #     # period_start -> time_idx 매핑(0부터 시작)
# #     def get_period_start(click_time, interval_minutes=1440, start_time=datetime.time(8, 0, 2)):

# #         base_start = datetime.datetime.combine(click_time.date(), start_time)
# #         if click_time < base_start:
# #             base_start -= datetime.timedelta(days=1)
# #         delta = click_time - base_start
# #         periods = int(delta.total_seconds() // (interval_minutes * 60))

# #         return base_start + datetime.timedelta(minutes=interval_minutes * periods)

# #     train_df['click_time'] = pd.to_datetime(train_df['click_time'])
# #     train_df['Period_Start'] = train_df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=30))
        
# #     unique_period_starts = train_df['Period_Start'].unique()
# #     time_dict = {ps: i+1680 for i, ps in enumerate(sorted(unique_period_starts))}
# #     train_df['time_idx'] = train_df['Period_Start'].map(time_dict)

# #     """
# #     train_news: 각 요소(리스트)는 train data에서 각 유저가 클릭한 news_ids
# #     - shape: (user_num, train data에서 각 유저의 클릭 수)

# #     train_category: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 categories
# #     - shape: (user_num, train data에서 각 유저의 클릭 수)

# #     train_time: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 times (snapshot or timestamp indicies)
# #     - shape: (user_num, train data에서 각 유저의 클릭 수)
# #     """
# #     train_news = []
# #     for u_id in tqdm(range(len(all_user_ids))):
# #         u_news = torch.tensor(train_df[train_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
# #         train_news.append(u_news)
        
# #     train_category = []
# #     for u_id in tqdm(range(len(all_user_ids))):
# #         u_category = torch.tensor(train_df[train_df['user_int'] == u_id]['cat_int'].values, dtype=torch.long)
# #         train_category.append(u_category)

# #     train_time = []
# #     for u_id in tqdm(range(len(all_user_ids))):
# #         u_time = torch.tensor(train_df[train_df['user_int'] == u_id]['time_idx'].values, dtype=torch.long)
# #         train_time.append(u_time)
        
# #     # print(train_time[0])
# #     # print(len(train_time[0]))


# #     # with open('./psj/Adressa_4w/train/train_datas.pkl', 'wb') as f:
# #     #     pickle.dump(list(zip(train_news, train_category, train_time)), f)
    
# #     return list(zip(train_news, train_category, train_time))


# import pickle
# import pandas as pd
# import torch
# from tqdm import tqdm
# import os


# def make_test_datas(snapshots_num: int):

#     # news2int 가져오기
#     news2int_file_path = '/home/user/pyo/psj/Adressa_4w/history/news2int.tsv'
#     news2int = pd.read_csv(news2int_file_path, sep='\t')

#     # a) test dataset(0212 08:00:02 ~ 0219 08:00:01)인 valid_tkg_behaviors.tsv 로드
#     test_file_path = '/home/user/pyo/psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
#     test_df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
    
#     # 'clicked_news' 열에서 '-1' 제거
#     test_df['clicked_news'] = test_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

#     # test_df에서 nan이 존재하는 행 제거
#     test_df = test_df.dropna(subset=['clicked_news'])



#     ########################################### 여기부터 negative sampling
#     # news2int를 dictionary로 변환
#     news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
#     file_path = '/home/user/pyo/psj/Adressa_4w/history/history_tkg_behaviors.tsv'
#     history_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
#     history_df['click_time'] = pd.to_datetime(history_df['click_time'])
#     end_time = pd.Timestamp('2017-02-05 08:00:01')
#     history_df = history_df[history_df['click_time'] <= end_time]   # 정확히 5주 데이터만 사용하도록 필터링

#     user2int_df = pd.read_csv(os.path.join('/home/user/pyo/psj/Adressa_4w/history/', 'user2int.tsv'), sep='\t')
#     user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
#     all_user_ids = [i for i in range(len(user2int_df))]   # 0 ~ 84988

#     test_df['user_int'] = test_df['history_user'].map(user2int)
#     test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)
#     category2int = pd.read_csv('/home/user/pyo/category2int_nyheter_splitted.tsv', sep='\t')
#     # 필요시 category2int에 'No category' 추가
#     if 'No category' not in category2int['category'].values:
#         new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
#         category2int = pd.concat([new_row, category2int], ignore_index=True)
#     cat2int = category2int.set_index('category')['int'].to_dict()
    
#     ############# category가 nyheter이면 subcategory로 매핑, 그렇지 않으면 category로 매핑
#     def get_cat_int(row):
#         if row['category'] == 'nyheter':
#             # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
#             return cat2int.get(row['subcategory'], cat2int['No category'])
#         else:
#             return cat2int.get(row['category'], cat2int['No category'])

#     test_df['cat_int'] = test_df.apply(get_cat_int, axis=1)



#     ### validation_df와 test_df로 분할
#     criteria_time = pd.Timestamp('2017-02-15 20:00:01')
#     test_df['click_time'] = pd.to_datetime(test_df['click_time'])

#     validation_df = test_df[test_df['click_time'] <= criteria_time]
#     test_5d_df = test_df[test_df['click_time'] > criteria_time]

#     # print(len(test_5d_df['history_user'].unique()))
#     # print(len(test_5d_df['user_int'].unique()))
#     # print(len(test_5d_df))
#     # print(len(validation_df['history_user'].unique()))
#     # print(len(validation_df['user_int'].unique()))
#     # print(len(validation_df))
#     # exit()


#     """
#     test_news: 각 요소(리스트)는 test data에서 각 유저가 클릭한 news_ids
#     - shape: (user_num, test data에서 각 유저의 클릭 수)

#     test_time: 각 요소(리스트)는 test data에서 각 유저가 클릭한 뉴스의 times
#     - shape: (user_num, test data에서 각 유저의 클릭 수)
#     """
#     test_news = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_news = torch.tensor(test_5d_df[test_5d_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
#         test_news.append(u_news)

#     test_time = []
#     test_empty_check = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_len = len(test_5d_df[test_5d_df['user_int'] == u_id])
#         u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
#         test_time.append(u_time)
#         if u_len == 0:
#             test_empty_check.append(False)
#         else:
#             test_empty_check.append(True)
        
#     # print(test_time[0])
#     # print(len(test_time[0]))

#     validation_news = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_news = torch.tensor(validation_df[validation_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
#         validation_news.append(u_news)

#     validation_time = []
#     validation_empty_check = []
#     for u_id in tqdm(range(len(all_user_ids))):
#         u_len = len(validation_df[validation_df['user_int'] == u_id])
#         u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
#         validation_time.append(u_time)
#         if u_len == 0:
#             validation_empty_check.append(False)
#         else:
#             validation_empty_check.append(True)

#     # print(validation_time[0])
#     # print(len(validation_time[0]))

#     # # 데이터 저장
#     # with open('./psj/Adressa_4w/test/test_datas.pkl', 'wb') as f:
#     #     pickle.dump(list(zip(test_news, test_time, test_empty_check)), f)
        
#     # with open('./psj/Adressa_4w/test/validation_datas.pkl', 'wb') as f:
#     #     pickle.dump(list(zip(validation_news, validation_time, validation_empty_check)), f)
    
#     return list(zip(validation_news, validation_time, validation_empty_check)), list(zip(test_news, test_time, test_empty_check))