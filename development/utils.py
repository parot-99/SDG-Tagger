def parse_sdg_id(sdg_id):
    if sdg_id < 0 or sdg_id > 15:
        raise Exception('No such SDG\n')

    names = {
        0: 'No Poverty',
        1: 'Zero Hunger',
        2: 'Good Health and Well-Being',
        3: 'Quality Education',
        4: 'Gender Equality',
        5: 'Clean Water and Sanitation',
        6: 'Affordable and Clean Energy',
        7: 'Decent Work and Economic Growth',
        8: 'Industry, Innovation, and Infrastructure',
        9: 'Reduced Inequalities',
        10: 'Sustainable Cites and Communities',
        11: 'Responsible Consumption and Production',
        12: 'Climate Action',
        13: 'Life Below Water',
        14: 'Life on Land',
        15: 'Peace, Justice, and Strong Institutions'
    }

    return names[sdg_id]

def parse_sdg_label(sdg_label):
    names = {
        'No Poverty': 0,
        'Zero Hunger': 1,
        'Good Health and Well-Being': 2,
        'Quality Education': 3,
        'Gender Equality': 4,
        'Clean Water and Sanitation': 5,
        'Affordable and Clean Energy': 6,
        'Decent Work and Economic Growth': 7,
        'Industry, Innovation, and Infrastructure': 8,
        'Reduced Inequalities': 9,
        'Sustainable Cites and Communities': 10,
        'Responsible Consumption and Production': 11,
        'Climate Action': 12,
        'Life Below Water': 13,
        'Life on Land': 14,
        'Peace, Justice, and Strong Institutions': 15
    }

    # if sdg_label not in list(names.keys()):
    #     raise Exception('No such SDG\n')
    
    sdg_id = names[sdg_label]

    return sdg_id

def prase_sentiment_label(sentiment_label):
    if sentiment_label < 0 or sentiment_label > 2:
        raise Exception('No such sentiment\n')
    
    sentiments = {
        0: 'Positive',
        1: 'Negative',
        2: 'Neutral'
    }

    return sentiments[sentiment_label]