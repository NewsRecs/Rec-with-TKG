import pickle
import pandas as pd
import torch
from tqdm import tqdm


def make_test_datas():
    # 각 뉴스의 카테고리만 가져오기
    test_news_file_path = './psj/Adressa_5w/test/news.tsv'
    test_news_df = pd.read_csv(test_news_file_path, sep='\t', header=None)
    test_news_df.columns = ['newsId', 'category', 'subcategory', 'title', 'body', 'identifier', 'publish_time', 'click_time']
    sub_test_news_df = test_news_df[['newsId', 'category']]

    # news2int 가져오기
    news2int_file_path = './psj/Adressa_4w/history/news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    # a) test dataset(0212 08:00:02 ~ 0219 08:00:01)인 valid_tkg_behaviors.tsv 로드
    test_file_path = './psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
    test_df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
    # 'clicked_news' 열에서 '-1' 제거
    test_df['clicked_news'] = test_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
    # 'clicked_newsId'를 기준으로 'category' 매칭
    test_df = test_df.merge(sub_test_news_df, left_on='clicked_news', right_on='newsId', how='left')
    # test_df에서 nan이 존재하는 행 제거
    test_df = test_df.dropna(subset=['clicked_news'])



    ########################################### 여기부터 negative sampling
    # news2int를 dictionary로 변환
    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    users = test_df['history_user'].unique()
    all_user_ids = [i for i in range(len(users))]
    user2int = {uid: i for i, uid in enumerate(users)}
    test_df['user_int'] = test_df['history_user'].map(user2int)
    test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)
    category2int = pd.read_csv('./psj/Adressa_4w/history/category2int_pio.tsv', sep='\t')
    # 필요시 category2int에 'No category' 추가
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)
    cat2int = category2int.set_index('category')['int'].to_dict()
    test_df['cat_int'] = test_df['category'].map(cat2int)#.fillna(0)



    ### validation_df와 test_df로 분할
    criteria_time = pd.Timestamp('2017-02-15 20:00:01')
    test_df['click_time'] = pd.to_datetime(test_df['click_time'])

    validation_df = test_df[test_df['click_time'] <= criteria_time]
    test_5d_df = test_df[test_df['click_time'] > criteria_time]

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
        u_news = torch.tensor(test_5d_df[test_5d_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        test_news.append(u_news)

    test_time = []
    test_empty_check = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_len = len(test_5d_df[test_5d_df['user_int'] == u_id])
        u_time = torch.tensor([2015 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
        test_time.append(u_time)
        if u_len == 0:
            test_empty_check.append(False)
        else:
            test_empty_check.append(True)
        
    # print(test_time[0])
    # print(len(test_time[0]))

    validation_news = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(validation_df[validation_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        validation_news.append(u_news)

    validation_time = []
    validation_empty_check = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_len = len(validation_df[validation_df['user_int'] == u_id])
        u_time = torch.tensor([2015 for _ in range(u_len)], dtype=torch.long)   # train까지 포함한 snapshot 수는 2016개
        validation_time.append(u_time)
        if u_len == 0:
            validation_empty_check.append(False)
        else:
            validation_empty_check.append(True)

    # print(validation_time[0])
    # print(len(validation_time[0]))

    # # 데이터 저장
    # with open('./psj/Adressa_4w/test/test_datas.pkl', 'wb') as f:
    #     pickle.dump(list(zip(test_news, test_time, test_empty_check)), f)
        
    # with open('./psj/Adressa_4w/test/validation_datas.pkl', 'wb') as f:
    #     pickle.dump(list(zip(validation_news, validation_time, validation_empty_check)), f)
    
    return list(zip(validation_news, validation_time, validation_empty_check)), list(zip(test_news, test_time, test_empty_check))