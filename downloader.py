import csv, os, traceback, time, sys
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

# Global Variables
## Blacklist with image-only datasets, meta-datasets (e.g., cheatsheets), models, libraries
BLACKLIST = [
    'abdallahalidev_plantvillage-dataset',
    'adityajn105_flickr8k',
    'ahmedhamada0_brain-tumor-detection',
    'alxmamaev_flowers-recognition',
    'aman2000jaiswal_agriculture-crop-images',
    'aman2000jaiswal_testssss',
    'andrewmvd_car-plate-detection',
    'andrewmvd_face-mask-detection',
    'arjuntejaswi_plant-village',
    'arunrk7_surface-crack-detection',
    'ashishjangra27_face-mask-12k-images-dataset',
    'ashishsaxena2209_animal-image-datasetdog-cat-and-panda',
    'bachrr_covid-chest-xray',
    'biaiscience_dogs-vs-cats',
    'chetankv_dogs-cats-images',
    'dansbecker_5-celebrity-faces-dataset',
    'emmarex_plantdisease',
    'fanconic_skin-cancer-malignant-vs-benign',
    'iarunava_cell-images-for-detecting-malaria',
    'jakeshbohaju_brain-tumor',
    'jessicali9530_celeba-dataset',
    'jonathanoheix_face-expression-recognition-dataset',
    'jutrera_stanford-car-dataset-by-classes-folder',
    'kaushiksuresh147_data-visualization-cheat-cheats-and-resources',
    'kmader_skin-cancer-mnist-ham10000',
    'knightnikhil_cardefect',
    'moltean_fruits',
    'msambare_fer2013',
    'muratkokludataset_grapevine-leaves-image-dataset',
    'muratkokludataset_pistachio-image-dataset',
    'muratkokludataset_rice-image-dataset',
    'mylesoneill_tagged-anime-illustrations',
    'neha1703_movie-genre-from-its-poster',
    'noulam_tomato',
    'phylake1337_fire-dataset',
    'psycon_feynman-diagrams-csv-png',
    'puneet6060_intel-image-classification',
    'ryuuseikuhome_azuralane-ships-data',
    'sapal6_bird-speciestiny',
    'sartajbhuvaji_brain-tumor-classification-mri',
    'scolianni_mnistasjpg',
    'splcher_animefacedataset',
    'sriramr_fruits-fresh-and-rotten-for-classification',
    'surajiiitm_bccd-dataset',
    'thedagger_pokemon-generation-one',
    'timoboz_data-science-cheat-sheets',
    'timoboz_python-data-science-handbook',
    'tongpython_cat-and-dog',
    'tourist55_alzheimers-dataset-4-class-of-images',
    'vbookshelf_v2-plant-seedlings-dataset',
    'vin1234_detecting-sentiments-of-a-quote',
    'vishalsubbiah_pokemon-images-and-types',
    'wobotintelligence_face-mask-detection-dataset',
    'xainano_handwrittenmathsymbols',
    'nih-chest-xrays_data',
    'cdeotte_rapids',
    'omkargurav_face-mask-dataset',
    'akash2sharma_tiny-imagenet',
    'cashutosh_gender-classification-dataset',
    'prondeau_the-car-connection-picture-dataset',
    'shawon10_ckplus',
    'huanghanchina_pascal-voc-2012',
    'andrewmvd_road-sign-detection',
    'arpitjain007_dog-vs-cat-fastai',
    'ashokpant_devanagari-character-dataset',
    'ashwingupta3012_human-faces',
    'atulyakumar98_pothole-detection-dataset',
    'biancaferreira_african-wildlife',
    'cenkbircanoglu_comic-books-classification',
    'dagnelies_deepfake-faces',
    'dansbecker_hot-dog-not-hot-dog',
    'dheerajperumandla_drowsiness-dataset',
    'dorianlazar_medium-articles-dataset',
    'groffo_ads16-dataset',
    'karakaggle_kaggle-cat-vs-dog-dataset',
    'kaustubhb999_tomatoleaf',
    'ma7555_cat-breeds-dataset',
    'nabeelsajid917_covid-19-x-ray-10000-images',
    'playlist_men-women-classification',
    'qiriro_ieee-tac',
    'raddar_ricord-covid19-xray-positive-tests',
    'ruchi798_covid19-pulmonary-abnormalities',
    'shank885_knife-dataset',
    'shrutisaxena_yoga-pose-image-classification-dataset',
    'tahsin_cassava-leaf-disease-merged',
    'tawsifurrahman_tuberculosis-tb-chest-xray-dataset',
    'veeralakrishna_butterfly-dataset',
    'yuval6967_gpt2-pytorch',
    # 50-100
    'cdeotte_512x512-melanoma-tfrecords-70k-images',
    'cdeotte_melanoma-512x512',
    'cdeotte_jpeg-melanoma-512x512',
    'iafoss_grapheme-imgs-128x128',
    'saroz014_plant-disease',
    # What about this one? 
    'authman_pickled-crawl300d2m-for-kernel-competitions',
    # What about this one?
    'ipythonx_efficientnet-keras-noisystudent-weights-b0b7'
    'nazmul0087_ct-kidney-dataset-normal-cyst-tumor-and-stone',
    'theoviel_rsna-breast-cancer-512-pngs',
    'google-brain_inception-v3',
    'theblackmamba31_landscape-image-colorization',
    'abhishek_pretrained-models',
    'abhishek_distilbertbaseuncased',
    'abhishek_transformers'
]
## Kaggle API
api = KaggleApi()
api.authenticate()


def download_datasets(dataframe: pl.DataFrame, download_path: str, exceptions_filename: str, unzip_threhsold: int = 6 * 10**9) -> None:
    start_time = time.time()

    n_downloads = 0
    exceptions = []
    refs = dataframe['ref']
    n_refs = len(refs)
    print('Will download %d datasets to %s' % (n_refs, download_path))

    # Download each dataset
    for i, ref in enumerate(refs):
        folder = ref.replace('/', '_')
        path = os.path.join(download_path, folder)

        download_start_time = time.time()

        # Check dataset not not in blacklist and not already downloaded
        if folder not in BLACKLIST and not os.path.isdir(os.path.join(download_path, folder)):
            try:
                # Check dataset size
                n_bytes = dataframe.filter(pl.col('ref') == ref)['totalBytesNullable'][0]
                unzip = True
                # Should unzip if above memory threshold
                if n_bytes is not None:
                    n_gbytes = n_bytes / 10**9
                    if n_bytes >= unzip_threhsold:
                        unzip = False

                print("Started downloading '%s' of size %.2f GB" % (ref, n_gbytes))
                api.dataset_download_files(ref, path, force=False, quiet=True, unzip=unzip)

                download_end_time = time.time()
                print("%d/%d Finished downloading '%s' in %.2f s" % (i + 1, n_refs, ref, download_end_time - download_start_time))

                n_downloads += 1

            except Exception as e:
                tr = str(traceback.format_exc())
                exceptions.append((ref, e, tr))
                print("\nEXCEPTION: occured while downloading '%s'..." % folder)
                print(tr)
                print(e)
                
    end_time = time.time()
    print('Finished downloading %d datasets, took %.2f s' % (n_downloads, end_time - start_time))

    # Write exceptions to file
    n_exceptions = len(exceptions)
    print('\nNumber of exceptions: %d' % n_exceptions)
    if n_exceptions > 0:
        print("\nWriting exceptions to file %s" % exceptions_filename)
        dataframe_exceptions = pd.DataFrame(exceptions, columns=['ref', 'exception', 'traceback'])
        dataframe_exceptions.to_csv(exceptions_filename)


# datasets_group_100_sampled
# datasets_group_50_100_sampled
# datasets_group_50_sampled
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception('Missing arguments <input_file_path> and <download_folder>')

    input_file = sys.argv[1]
    if Path(input_file).suffix != '.csv':
        raise Exception('Input file must be in csv format')
    
    download_path = sys.argv[2]
    if not os.path.isdir(download_path):
        os.mkdir(download_path)

    downloaded_folders = next(os.walk(download_path))[1]
    print('Num datasets downloaded so far: %d' % len(downloaded_folders))

    # Load dataframe with information about datasets to download
    dataframe = pl.read_csv(input_file)

    # Check how many folders have been already downloaded
    n_already_downloaded = 0
    refs_to_download = dataframe['ref']
    for downloaded_folder in downloaded_folders:
        if downloaded_folder in [ref.replace('/', '_') for ref in refs_to_download]:
            n_already_downloaded += 1
    print('%d/%d datasets already downloaded from given refs' % (n_already_downloaded, len(refs_to_download)))

    exceptions_filename = 'exceptions_' + input_file
    # Download datasets
    download_datasets(dataframe, download_path, exceptions_filename)

    print('\nDONE')