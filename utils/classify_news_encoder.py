def get_news_encoder(config):
    method = config.method
    use_category = not config.no_category

    if method == 'cnn_attention':
        if use_category:
            from utils.full_news_encoder import NewsEncoder
        else:
            from utils.title_news_encoder import NewsEncoder
    else:  # method == 'multihead_self_attention'
        if use_category:
            from utils.MSA_news_encoder import NewsEncoder
        else:
            from utils.title_MSA_news_encoder import NewsEncoder

    return NewsEncoder
