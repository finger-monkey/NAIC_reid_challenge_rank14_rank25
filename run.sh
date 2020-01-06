# thanks a lot!
python convert2trainset.py # conver origin dataset to trainset
bash train_base.sh         # train base model
python get_violet_test.py  # split test_data
bash features.sh           # extract base model fetures
bash cluster.sh            # cluster origin pic to trainset
bash copy.sh               # copy to trainset
bash train_final.sh        # train final model
bash final_features.sh     # get final features
python reranking_submit.py # reranking
