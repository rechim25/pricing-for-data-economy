import csv
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import time
import itertools
import typer
from kaggle.api.kaggle_api_extended import KaggleApi

DOMAIN_TO_KAGGLE_TAGS = {
    # 'Manufacturing': ['Manufacturing'],
    'Automotive': ['Automobiles and Vehicles'],
    'Gaming': ['Games', 'Video Games'],
    'Healthcare': ['Biology', 'Healthcare', 'Health', 'Medicine'],
    'Retail & Marketing': ['Retail and Shopping', 'Travel'],
    'Resources': ['Earth and Nature', 'Earth Science', 'Atmospheric Science', 'Oil and Gas', 'Water Bodies', 'Energy', 'Electricity', 'Geology'],
    'Media & Entertainment': ['News', 'Social Networks', 'Arts and Entertainment'],
    'Financial': ['Finance', 'Economics', 'Investing', 'Insurance'],
    'Other': ['Demographics', 'Housing', 'Social Science', 'Education']
}
DOMAINS = list(DOMAIN_TO_KAGGLE_TAGS.keys())

TAGS = list(itertools.chain.from_iterable(DOMAIN_TO_KAGGLE_TAGS.values()))

# tag_to_dataset_count = {}
# for tag in TAGS:
#     tag_to_dataset_count[tag] = 0

api = KaggleApi()
api.authenticate()

app = typer.Typer()


def get_tag_info_from_first_page(tag_id: str):
    # Get datasets on first page (sorted by votes)
    datasets = api.dataset_list(tag_ids=tag_id, sort_by='votes')
    for dataset in datasets:
        for tag_obj in dataset.tags:
            if tag_obj.nameNullable == tag_id.lower() or tag_obj.ref == tag_id.lower():
                return [tag_obj.ref, tag_obj.fullPath, tag_obj.datasetCount]
    return []


def get_tag_infos():
    TAGS_FIELDS = ['ref', 'fullPath', 'datasetCount']
    with open('tags.csv', 'w') as tags_file:
        # Write fields on first row of CSV file
        writer_tags = csv.writer(tags_file, delimiter=',')
        writer_tags.writerow(TAGS_FIELDS)
        for domain in DOMAINS:
            for tag_id in DOMAIN_TO_KAGGLE_TAGS[domain]:
                tag_info_list = get_tag_info_from_first_page(tag_id)
                if len(tag_info_list):
                    writer_tags.writerow(tag_info_list)
                else:
                    raise Exception("Tag info not found on first page for tag_id=" + str(tag_id))
                
# tag_ids: list[str] = typer.Argument(TAGS),        
@app.command() 
def get_datasets_csv(filename: str):
    DATASETS_FIELDS = ['index', 'ref', 'voteCount', 'usabilityRatingNullable', 'downloadCount', 
                       'totalBytesNullable', 'lastUpdated', 'tags', 'used_tag']
    print("Grabbing dataset info from Kaggle API...")
    print("\n %d Tags: %s" % (len(TAGS), str(TAGS)))
    print("\n")
    n_datasets_total = 0
    start_time = time.time()
    with open(filename + '.csv', 'w') as datasets_file:
        writer_datasets = csv.writer(datasets_file, delimiter=',')
        # Write fields on first row of CSV file
        writer_datasets.writerow(DATASETS_FIELDS)
        for n, tag_id in enumerate(TAGS):
            page = 1
            n_datasets_per_tag = 0
            while True: # Pagination
                datasets = api.dataset_list(tag_ids=tag_id, sort_by='votes', file_type='csv', max_size=10**7, page=page)
                n_datasets_per_tag += len(datasets)
                page += 1
                if len(datasets) > 0:
                    # Read all datasets from page
                    for i, ds in enumerate(datasets):
                        writer_datasets.writerow([i, ds.ref, ds.voteCount, ds.usabilityRatingNullable, ds.downloadCount, 
                                                ds.totalBytesNullable, ds.lastUpdated, ds.tags, tag_id])   
                else:
                    # Encountered empty page (no more datasets)
                    print("%d/%d Found %d datasets (%d pages) for tag_id=%s" % (n + 1, len(TAGS), n_datasets_per_tag, page, tag_id))
                    n_datasets_total += n_datasets_per_tag
                    break
    end_time = time.time()
    print("Took %f s. Found %d datasets in total" % (end_time - start_time, n_datasets_total))

if __name__ == '__main__':
    app()