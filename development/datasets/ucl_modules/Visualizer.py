import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, data):
        self.__data = data
        self.__colors = [
            'blue',
            'orange',
            'green',
            'red',
            'purple',
            'brown',
            'pink',
            'gray',
            'olive',
            'cyan',
            'olive',
            'yellow',
            'crimson',
            'slateblue',
            'darkgreen',
            'tan',
        ]

    def print_info(self):
        print(f'sdg_counts: A barplot showing the total number of taught modules per SDG')

    # visualizers

    def sdgs_count(self):
        sdg_names = [f'SDG {i}' for i in range(1, 17)]
        sdgs_counts = [self.__data[f'SDG_{i}'].sum() for i in range(1, 17)]

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Number of SDGs-related taught modules across UCL')
        ax.set_xlabel('Number of modules')
        ax.set_xticks([i for i in range(0, 3001, 250)])
        ax.xaxis.grid(
            True,
            linestyle='--',
            which='major',
            color='grey',
            alpha=.25
        )
        bar_plot = ax.barh(
            sdg_names,
            sdgs_counts,
            align='center',
            color=self.__colors
        )

        plt.show()

    def sdgs_count_faculty(self, faculty_name):
        faculty_data = self.__data[self.__data['Faculty'] == faculty_name]
        sdg_names = [f'SDG {i}' for i in range(1, 17)]
        sdgs_counts = [faculty_data[f'SDG_{i}'].sum() for i in range(1, 17)]

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Number of SDGs-related taught modules by SDG by department')
        ax.set_xlabel(f'Number of modules in {faculty_name}')
        ax.set_xticks([i for i in range(0, 701, 50)])
        ax.xaxis.grid(
            True,
            linestyle='--',
            which='major',
            color='grey',
            alpha=.25
        )
        bar_plot = ax.barh(
            sdg_names,
            sdgs_counts,
            align='center',
            color=self.__colors
        )

        plt.show()

    def sdg_nosdg_pie(self):
        sdg_modules = self.__data[~self.__data['final_sdg_labels'].isna()]
        sdg_modules = sdg_modules.count().max()
        no_sdg_modules = self.__data[self.__data['final_sdg_labels'].isna()]
        no_sdg_modules = no_sdg_modules.count().max()
        labels = ['SDG Related Modules', 'Non-SDG Related Modules']

        # plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Title')
        ax.pie(
            [sdg_modules, no_sdg_modules],
            labels=labels,
            autopct='%1.1f%%',
            colors=['blue', 'red'],
            explode=[0, 0.2],
            shadow=True
        )
        plt.plot()

