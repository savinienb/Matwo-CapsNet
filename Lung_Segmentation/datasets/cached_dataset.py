
from datasets.dataset_base import DatasetBase
import threading
import multiprocessing
import random


class CachedDataset(DatasetBase):
    """
    Dataset that prefetches and caches entry dicts for an encapsulated dataset (self.dataset).
    Uses an internal queue that caches previous entry dicts, while get() returns a random
    entry of the internal queue. If the queue is full and a new entry dict is calculated, the
    oldest entry of the internal queue gets removed.
    """
    def __init__(self, dataset, n_threads=32, queue_size=512, start_threads=True):
        """
        Initializer.
        :param dataset: The wrapped dataset.
        :param n_threads: The number of threads.
        :param queue_size: The maximum size of the queue.
        :param start_threads: If true, the prefetching threads are started right away.
                              Otherwise, start_threads() must be called to start the prefetching threads.
        """
        self.should_stop = False
        self.queue_size = queue_size
        self.queue = []
        self.dataset = dataset
        self.n_threads = n_threads
        self.threads = []
        self.lock = multiprocessing.Lock()
        if start_threads:
            self.start_threads()

    def __del__(self):
        """
        Destructor, stops the threads.
        """
        self.stop_threads()

    def thread_main(self):
        """
        Main function of the prefetching threads.
        """
        print('CachedDataset thread start')
        while not self.should_stop:
            entry = self.dataset.get_next()
            with self.lock:
                self.queue.append(entry)
                if len(self.queue) > self.queue_size:
                    self.queue.pop(0)
        print('CachedDataset thread stop')

    def start_threads(self):
        """
        Starts the prefetching threads.
        """
        self.should_stop = False
        for _ in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main)
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

    def stop_threads(self):
        """
        Stops and joins the prefetching threads.
        """
        self.should_stop = True
        self.queue.clear()
        for thread in self.threads:
            thread.join()

    def get_next(self):
        """
        Returns a random next datasets entry of the internal queue.
        """
        with self.lock:
            index = random.randrange(0, len(self.queue))
            return self.queue[index]

    def num_entries(self):
        """
        Not supported.
        """
        raise RuntimeError('num_entries() is not supported for CachedDataset.')

    def get(self, id_dict):
        """
        Not supported.
        """
        raise RuntimeError('get(id_dict) is not supported for CachedDataset. Use get_next() instead.')
