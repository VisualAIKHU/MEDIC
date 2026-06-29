def create_dataset(cfg, split='train'):
    dataset = None
    data_loader = None
    
    if cfg.data.dataset == 'rcc_dataset_transformer_type_aware_chg':
        from datasets.rcc_dataset_transformer_type_aware_chg import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)
    elif cfg.data.dataset == 'rcc_dataset_transformer_type_aware_dc':
        from datasets.rcc_dataset_transformer_type_aware_dc import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)
    elif cfg.data.dataset == 'rcc_dataset_transformer_type_aware_std':
        from datasets.rcc_dataset_transformer_type_aware_std import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)
    elif cfg.data.dataset == 'rcc_dataset_transformer_type_aware_ier':
        from datasets.rcc_dataset_transformer_type_aware_ier import RCCDataset, RCCDataLoader
        dataset = RCCDataset(cfg, split)
        data_loader = RCCDataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers,
            pin_memory=True)

    return dataset, data_loader
