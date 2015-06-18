import requests
import csv
import logging

base_url = '/root/anaconda/envs/cv-project/project/data/facescrub/'
image_url = '/root/anaconda/envs/cv-project/project/data/facescrub/image/'

# configure logging
logging.basicConfig(filename=base_url + 'scraper.log',level=logging.INFO)

# read tsv file
with open(base_url+'facescrub_actors.txt', 'rb') as f:
    csvfile = csv.reader(f, delimiter='\t')

    # skip header
    next(csvfile)

    # counter
    success = 0
    error = 0

    logging.info('scraping actor images')

    # read from url
    for row in csvfile:
        url = row[3]

        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                full_path = '%s%s%s%s' % (image_url, 'actor_', row[2], '.jpg')
                with open(full_path, 'rb') as image_file:
                    for chunk in r.iter_content():
                        image_file.write(chunk)

                    success += 1
                    logging.info('successfully scrape actor image ' + row[2])

        except Exception:
            logging.exception('error scraping actor image ' + row[2])
            error += 1

    logging.info('actor image: success: %d, error: %d' % (success, error))
