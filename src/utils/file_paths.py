# src/utils/file_paths.py

from pathlib import Path



class FilePath:
    def __init__(self):
        # Base path (assuming this file is in src/utils/)
        self.base_path = Path(__file__).parent.parent.parent

        # Data paths
        self.data_path = self.base_path / 'data'

        # Raw data paths
        self.raw_path = self.data_path / 'raw'
        self.cervix_raw_path = self.raw_path / 'cervix'
        self.brain_raw_path = self.raw_path / 'brain'
        self.afmmm_raw_path = self.raw_path / 'afmmm'

        # Interim data paths
        self.interim_path = self.data_path / 'interim'
        self.cervix_interim_path = self.interim_path / 'cervix'
        self.brain_interim_path = self.interim_path / 'brain'
        self.afmmm_interim_path = self.interim_path / 'afmmm'

        self.combined_interim_path = self.interim_path / 'combined'

        # Processed data paths
        self.processed_path = self.data_path / 'processed'
        self.cervix_processed_path = self.processed_path / 'cervix'
        self.brain_processed_path = self.processed_path / 'brain'
        self.afmmm_processed_path = self.processed_path / 'afmmm'

        # Results paths
        self.results = self.base_path / 'results'
        self.figures = self.results / 'figures'

        # Model path
        self.model_save_path = self.base_path / 'model'


# Create an instance of the FilePath class
file_paths = FilePath()



TISSUE_DIMENSIONS = {
    'cervix': {'num_rows': 600, 'num_cols': 800},
    'brain': {'num_rows': 388, 'num_cols': 516},
    'afmmm': {'num_rows': 500, 'num_cols': 500}
}