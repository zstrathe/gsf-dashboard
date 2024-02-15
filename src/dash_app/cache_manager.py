import os
import shutil

class FileCache:
    """
    A class for caching files on local storage.

    Attributes:
        cache_dir (str): The directory path where cached items are stored.
    """

    def __init__(self, cache_dir:str):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def add_cache_item(self, cache_name, data):
        """
        Adds an item to the cache directory.

        params:
            cache_name (str): Identifier for the item to be cached
            data (bytes): The binary data of the item

        Returns:
            None
        """
        filepath = os.path.join(self.cache_dir, cache_name)
        with open(filepath, 'wb') as f:
            f.write(data)

    def get_cache_item(self, cache_name):
        """
        Retrieves an item from the cache directory.

        params:
            cache_name (str): identifier for the cache item

        Returns:
            bytes: The binary data of the cached item, or None if it is not found.
        """
        filepath = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return f.read()
        else:
            print(f"{cache_name} not found in cache.")
            return None

    def delete_cache_item(self, cache_name):
        """
        Deletes an item from the cache directory.

        params:
            cache_name (str): identifier for the cache item

        Returns:
            None
        """
        filepath = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"{cache_name} deleted from cache.")
        else:
            print(f"{cache_name} not found in cache.")

    def clear_cache(self):
        """
        Clears the entire cache directory.

        Returns:
            None
        """
        shutil.rmtree(self.cache_dir)
        print("Cache cleared.")