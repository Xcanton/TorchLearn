# Iterable-style Datasets

&ensp;&ensp;&ensp;&ensp;An iterable-style dataset is an instance of a subclass of [ <b>_`IterableDataset`_</b> ](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) that implements the `__iter__()` protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.

&ensp;&ensp;&ensp;&ensp;For example, such a dataset, when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time.

## Build-in Distribute Strategy

&ensp;&ensp;&ensp;&ensp; Index can be seperated into several groups according to the number of workers manully. This can be defined in the `__iter__()` function.

```
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

# Mult-process loading with two worker processes
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
```

## Outer function Distribute Strategy

&ensp;&ensp;&ensp;&ensp; Also, this can be defined via a outer function.

```
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=7)

# Single-process loading
print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
# Directly doing multi-process loading yields duplicate data
print(list(torch.utils.data.DataLoader(ds, num_workers=2)))

# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))

# With even more workers
print(list(torch.utils.data.DataLoader(ds, num_workers=12, worker_init_fn=worker_init_fn)))
```

# Reference
[1] [https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)</br>