def parse_label(sdg_label):
    if sdg_label < 1 or sdg_label > 16:
        raise Exception('No such SDG\n')

    names = {
        1: 'No Poverty',
        2: 'Zero Hunger',
        3: 'Good Health and Well-Being',
        4: 'Quality Education',
        5: 'Gender Equality',
        6: 'Clean Water and Sanitation',
        7: 'Affordable and Clean Energy',
        8: 'Decent Work and Economic Growth',
        9: 'Industry, Innovation, and Infrastructure',
        10: 'Reduced Inequalities',
        11: 'Sustainable Cites and Communities',
        12: 'Responsible Consumption and Production',
        13: 'Climate Action',
        14: 'Life Below Water',
        15: 'Life on Land',
        16: 'Peace, Justice, and Strong Institutions'
    }

    return names[sdg_label]