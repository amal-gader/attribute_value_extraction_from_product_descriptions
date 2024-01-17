import pickle
import os

cache_dir = os.environ.get('CACHE_DIR')
os.makedirs(cache_dir, exist_ok=True)


def cache_data(data_loader):
    def wrapper(*args, force_update=False, **kwargs):
        cache_key = f"{data_loader.__name__}.pkl"
        # Check if cache file exists and force_update is False
        cache_file = os.path.join(cache_dir, cache_key)
        if not force_update and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                cached_data = pickle.load(file)
            print(f"Loaded data from cache: {cache_key}")
            return cached_data
        else:
            # Call the data loader function to load data
            data = data_loader(*args, **kwargs)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as file:
                pickle.dump(data, file)
            print(f"Saved data to cache: {cache_key}")
            return data

    return wrapper
