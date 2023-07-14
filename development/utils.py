def parse_label(sdg_label):
    if sdg_label < 0 or sdg_label > 15:
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

    return names[sdg_label]