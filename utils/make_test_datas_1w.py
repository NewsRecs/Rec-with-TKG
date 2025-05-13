import pickle
import pandas as pd
import torch
from tqdm import tqdm
import os


def make_test_datas(snapshots_num: int):

    # news2int 가져오기
    news2int_file_path = '/home/user/pyo/psj/Adressa_4w/history/news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    # a) test dataset(0212 08:00:02 ~ 0219 08:00:01)인 valid_tkg_behaviors.tsv 로드
    test_file_path = '/home/user/pyo/psj/Adressa_1w/train/history_tkg_behaviors.tsv'
    df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
    # click_time을 string에서 datetime으로 변환
    df['click_time'] = pd.to_datetime(df['click_time'])
    
    criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
    test_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    # test_df에서 nan이 존재하는 행 제거
    test_df = test_df.dropna(subset=['clicked_news'])



    ########################################### 여기부터 negative sampling
    # news2int를 dictionary로 변환
    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
    # user2int mapping
    user2int_df = pd.read_csv(os.path.join('/home/user/pyo/psj/Adressa_1w/datas/', 'user2int.tsv'), sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    all_user_ids = user2int_df['user_int'].tolist()   # 0 ~ 84988

    test_df['user_int'] = test_df['history_user'].map(user2int)
    test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)
    category2int = pd.read_csv('/home/user/pyo/category2int_nyheter_splitted.tsv', sep='\t')
    # 필요시 category2int에 'No category' 추가
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)
    cat2int = category2int.set_index('category')['int'].to_dict()
    
    ############# category가 nyheter이면 subcategory로 매핑, 그렇지 않으면 category로 매핑
    def get_cat_int(row):
        if row['category'] == 'nyheter':
            # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
            return cat2int.get(row['subcategory'], cat2int['No category'])
        else:
            return cat2int.get(row['category'], cat2int['No category'])

    test_df['cat_int'] = test_df.apply(get_cat_int, axis=1)



    # ### validation_df와 test_df로 분할
    # criteria_time = pd.Timestamp('2017-02-15 20:00:01')
    # test_df['click_time'] = pd.to_datetime(test_df['click_time'])

    # validation_df = test_df[test_df['click_time'] <= criteria_time]
    # test_5d_df = test_df[test_df['click_time'] > criteria_time]

    # print(len(test_5d_df['history_user'].unique()))
    # print(len(test_5d_df['user_int'].unique()))
    # print(len(test_5d_df))
    # print(len(validation_df['history_user'].unique()))
    # print(len(validation_df['user_int'].unique()))
    # print(len(validation_df))
    # exit()


    """
    test_news: 각 요소(리스트)는 test data에서 각 유저가 클릭한 news_ids
    - shape: (user_num, test data에서 각 유저의 클릭 수)

    test_time: 각 요소(리스트)는 test data에서 각 유저가 클릭한 뉴스의 times
    - shape: (user_num, test data에서 각 유저의 클릭 수)
    """
    test_news = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(test_df[test_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        test_news.append(u_news)

    test_time = []
    test_empty_check = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_len = len(test_df[test_df['user_int'] == u_id])
        u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
        test_time.append(u_time)
        if u_len == 0:
            test_empty_check.append(False)
        else:
            test_empty_check.append(True)
        
    # print(test_time[0])
    # print(len(test_time[0]))

    # validation_news = []
    # for u_id in tqdm(range(len(all_user_ids))):
    #     u_news = torch.tensor(validation_df[validation_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
    #     validation_news.append(u_news)

    # validation_time = []
    # validation_empty_check = []
    # for u_id in tqdm(range(len(all_user_ids))):
    #     u_len = len(validation_df[validation_df['user_int'] == u_id])
    #     u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
    #     validation_time.append(u_time)
    #     if u_len == 0:
    #         validation_empty_check.append(False)
    #     else:
    #         validation_empty_check.append(True)

    # # print(validation_time[0])
    # # print(len(validation_time[0]))

    # # # 데이터 저장
    # # with open('./psj/Adressa_4w/test/test_datas.pkl', 'wb') as f:
    # #     pickle.dump(list(zip(test_news, test_time, test_empty_check)), f)
        
    # # with open('./psj/Adressa_4w/test/validation_datas.pkl', 'wb') as f:
    # #     pickle.dump(list(zip(validation_news, validation_time, validation_empty_check)), f)
    
    return list(zip(test_news, test_time, test_empty_check))