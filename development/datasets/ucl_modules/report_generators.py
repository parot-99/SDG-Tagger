from pandas import read_excel, ExcelWriter, DataFrame


def generate_reports(data_path, append=False, report_names=[]):
    reports = []

    report_generators = [
        report_faculty_sdgs,
        report_faculty_total_sdgs
    ]

    for i, report_generator in enumerate(report_generators):
        report = report_generator(
            data_path,
            append=append,
            report_name=report_names[i]
        )
        reports.append(report)

    return reports

def append_report(data_path, report, report_name):
    with ExcelWriter(data_path, mode='a') as writer:
        report.to_excel(writer, sheet_name=report_name, index=False)

# report generators

def report_faculty_sdgs(data_path, append=False, report_name='Report Name'):
    data = read_excel(data_path, sheet_name='Data')
    report = DataFrame(columns=[
        'Faculty',
        'SDG Courses',
        'Total # of Courses',
        'Percentage'
    ])
    
    modules_by_faculty = data.groupby(['Faculty'])['Faculty'].count()
    sdg_courses = data[data['final_sdg_labels'] != '']
    sdg_courses_count = sdg_courses.groupby(['Faculty'])['final_sdg_labels'].count().values


    report['Faculty'] = modules_by_faculty.index
    report['Total # of Courses'] = modules_by_faculty.values
    report['SDG Courses'] = sdg_courses_count

    percentage = report['SDG Courses'] * 100 / report['Total # of Courses']
    report['Percentage'] = percentage
    report['Percentage'] = report['Percentage'].apply(lambda x: f'{x:.2f}%')

    percentage =  report['SDG Courses'].sum() * 100 / report['Total # of Courses'].sum()
    report.loc[report.shape[0]] = [
        'All Faculties',
        report['SDG Courses'].sum(),
        report['Total # of Courses'].sum(),
        f'{percentage:.2f}%'
    ]

    if append:
        append_report(data_path, report, report_name)

    return report


def report_faculty_total_sdgs(data_path, append=False, report_name='Report'):
    data = read_excel(data_path, sheet_name='Data')
    report = DataFrame(columns=[
        'Faculty',
        *[f'SDG_{i}' for i in range(1, 17)],
        'No SDGs'
    ])

    # comment
    faculty_names = data.groupby(['Faculty'])['Faculty'].count().index
    report['Faculty'] = faculty_names

    # comment
    for i in range(1, 17):
        column = data.groupby('Faculty')[f'SDG_{i}'].sum()
        report[f'SDG_{i}'] = column.values

    # comment
    modules_no_sdgs = []

    for faculty in faculty_names:
        no_sdgs = data[
            (data['Faculty'] == faculty) & (data['final_sdg_labels'].isna())
        ]
        modules_no_sdgs.append(
            no_sdgs.count().max()
        )

    report['No SDGs'] = modules_no_sdgs

    # comment
    report.loc[report.shape[0]] = [
        'Total no of UCL courses addressing this SDG:',
        *[f'{report[f"SDG_{i}"].sum()}' for i in range(1, 17)],
        report['No SDGs'].sum()
    ]

    if append:
        append_report(data_path, report, report_name)

    return report