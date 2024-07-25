import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from s3_functions import download_file


download_file('airplane_schedule.csv', 'data/airplane_schedule.csv')
download_file('val_dataset.csv', 'data/val_dataset.csv')