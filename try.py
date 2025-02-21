datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
        ]