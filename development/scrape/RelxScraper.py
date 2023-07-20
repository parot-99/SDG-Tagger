import requests
import pandas as pd
import logging
from bs4 import BeautifulSoup

class RelxScraper:
    def __init__(self):
        self.__data = pd.DataFrame(columns=[
            'title',
            'link',
            'abstract',
            'sdgs',
            'full_abstract'
        ])
        self.__base_url = 'https://sdgresources.relx.com'
        self.__url = 'https://sdgresources.relx.com/articles/climate-change-impacts-water-security-global-drylan'
        self.__num_pages = 291

    def scrape_data(self, verbose=1, start=0):
        for page_num in range(start, self.__num_pages):
            logging.basicConfig(level=logging.CRITICAL)

            if verbose == 1:
                logging.info(f'Scraping page number {page_num + 1}')
                print(f'Scraping page number {page_num + 1}')

            if page_num != 0:
                page = requests.get(self.__url + f'?page={page_num}')
                soup = BeautifulSoup(page.content, 'html.parser')

            else:
                page = requests.get(self.__url)
                soup = BeautifulSoup(page.content, 'html.parser')

            articles = soup.find_all('div', class_='views-row')

            for article in articles:
                article_data = self.__get_all_data(article)

                if article_data is not None:
                    self.__data.loc[len(self.__data),:] = article_data

            if verbose == 1:
                logging.info(f'Scraping page number {page_num + 1} finished')

        self.__add_sdg_columns()

    def __get_all_data(self, article):
        title = article.find('h2', class_='field-content')
        title = title.find('a').text
        link = article.find('h2', class_='field-content')
        link = link.find('a')['href']
        abstract = article.find('div', class_='views-field-body')

        if abstract is None:
            return None
        
        abstract = abstract.find('div').text
        sdgs = article.find('div', class_='sidebar-content')
        sdgs = sdgs.find('div', class_='field-content')

        if sdgs is None:
            return None
        
        sdgs = sdgs.text
        full_abstract = self.__get_full_abstract(link)

        if full_abstract is not None:
            new_row = {
                'title': title,
                'link': link,
                'abstract': abstract,
                'sdgs': sdgs,
                'full_abstract': full_abstract
            }

            return new_row
        
        else:
            return None

    def __get_full_abstract(self, link):
        full_abstract_page = requests.get(self.__base_url + link)
        full_abstract_page = BeautifulSoup(
            full_abstract_page.content,
            'html.parser'
        )
        full_abstract = full_abstract_page.find(
            'div', class_='field-type-text-with-summary'
        )

        try:
            full_abstract = full_abstract.find('p').text

        except AttributeError as e:
            full_abstract = None

        return full_abstract
    
    def __add_sdg_columns(self):
        for i in range(1, 17):
            self.__data[f'sdg_{i}'] = 0.0

        def in_sdgs(sdgs, i):
            sdgs_list = []
                
            for sdg in sdgs['sdgs'].split(';'):
                label = sdg.strip().split(' ')[1].split(':')[0]
                sdgs_list.append(int(label))

            if i in sdgs_list:
                return 1.0
 
            return 0.0

        for i in range(1, 17):
            self.__data[f'sdg_{i}'] = self.__data.apply(
            lambda x: in_sdgs(x, i),
            axis=1
        )

    def save_as_csv(self, path):
        self.__data.to_csv(path, sep=',', index=False, encoding='utf-8')

    # properties

    @property
    def data(self):
        if self.__data.empty:
            raise Exception('Data not scraped yet; use RelxScraper.scrape()')

        return self.__data
    
