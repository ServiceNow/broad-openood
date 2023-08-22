This is a fork of the [`OpenOOD`](https://github.com/Jingkang50/OpenOOD) repository, containing additions to reproduce experiments made in the paper Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection.

For installation, please refer to the [original repository](https://github.com/Jingkang50/OpenOOD).

Additions include CADet postprocessor, GMM ensemble postprocessor, and the store_stats pipeline which is used to compute and store scores, as well as a number of bug fixes (the repository now correctly supports ViT) and some additional features.
Examples of config files are provided in `configs/postprocessors/cadet_postprocessor.yml` and `configs/postprocessors/gmm_ens_postprocessor.yml`.

For the gmm ensemble postprocessor, first make sure all scores have been saved in `config.stats_dir/network_name/dataset_name/statistic_name`. 
The model will be trained on the scores provided for the dataset `config.postprocessor.postprocessor_args.id_ds_name` and will compute stats for the datasets in `config.ood_dataset.ood.datasets`.
