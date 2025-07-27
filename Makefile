.PHONY: env install clean metadata

env:
	conda create -n bio python==3.11.*

setup:
	pip install poetry==2.1.3
	poetry install --no-root

clean:
	conda env remove -n bio

metadata:
	python ./src/metadata.py -i ./data/polyommatus_scans_and_photos_v3/common_group/ -o ./data -f meta_polyommatus_scans_and_photos_v3_common_group
	python ./src/metadata.py -i ./data/polyommatus_scans_and_photos_v3/target_group/ -o ./data -f meta_polyommatus_scans_and_photos_v3
	python -c "import pandas as pd;df_cmn = pd.read_csv('./data/meta_polyommatus_scans_and_photos_v3.csv');df_tgt = pd.read_csv('./data/meta_polyommatus_scans_and_photos_v3_common_group.csv');df_all = pd.concat([df_cmn, df_tgt]);df_all.to_csv('./data/meta_all_groups_v3.csv', index=False)"
