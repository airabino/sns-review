import requests

from bs4 import BeautifulSoup

from .progress_bar import ProgressBar

def extract(data):

    prices = []

    for car in data:

        prices.append(int(car['pricing']['0:  Germany'][1:].replace(',', '')))

    out = {}
    out['prices'] = prices

    return out

def veh_name(data):

    data['name'] = data['url'].split('/')[-1]

    return data

def parse_table(data, soup, title):

    section = soup.find(id = title)
    print(soup)

    tables = section.findAll("table")

    out = {}

    for idx_t, table in enumerate(tables):

        values = table.findAll('td')

        for idx in range(0, len(values), 2):

            out[f'{idx_t}: {values[idx].text}'] = values[idx + 1].text

    data[title] = out

    return data

def pull(url = "https://ev-database.org/cheatsheet/range-electric-car"):

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    return soup

def parse(soup):

    results = soup.find(id = "cheatsheet-page")
    tables = results.find_all("div", class_ = "core-content")
    table = tables[0]

    data = []

    idx = 0

    for element in ProgressBar(table.findAll("a")):

        # try:

        car = {'url': "https://ev-database.org" + element.get("href")}

        car = veh_name(car)

        car_soup = pull(car['url'])

        print(car)


        car = parse_table(car, car_soup, 'pricing')
        car = parse_table(car, car_soup, 'range')
        car = parse_table(car, car_soup, 'battery')
        car = parse_table(car, car_soup, 'charging')


        data.append(car)

        # except:

        #     pass

        # if idx >= 3:

        #     break

        # idx +=1
    
    return data





