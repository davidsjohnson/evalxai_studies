from pathlib import Path

import pandas as pd

FILTERS = ['study1-59', 'study1-60']
REJECTED = ['study1-53', 'study1-54', 'study1-55', 'study1-56', 'study1-57', 'study1-58', 'study1-59']

STAGES = {'1': 'validation_nodiag', '2': 'validation_noxai'}
STAKES_FOLDERS = {'high': 'example_validation_study',
                  'low': 'example_validation_study_ls'}

# new dataframe structure one dataframe per condition: 
# cols: part_id, blocky_id, y_true, y_pred, y_user_base, y_user_advised, time_taken_base, time_taken_advised

def _get_id(res_file: Path):
    return res_file.stem.split('_')[-1]

def process_participant_results(participant_id, stakes_level, stakes_path):
    """ extracts results from participant results file into dataframe """
    results_file1 = stakes_path / STAGES['1'] / f'results/results_{STAGES['1']}_{participant_id}.csv'
    results_file2 = stakes_path / STAGES['2'] / f'results/results_{STAGES['2']}_{participant_id}.csv'

    if results_file1.exists() and results_file2.exists():
        df_1 = pd.read_csv(results_file1)
        df_2 = pd.read_csv(results_file2)
        assert len(df_1) == len(df_2), "Mismatched lengths between stages"

        # ensure blocky IDs match to align user selections
        df_1 = df_1.sort_values(by='id').reset_index(drop=True)
        df_2 = df_2.sort_values(by='id').reset_index(drop=True)

        # drop rows with missing user selections
        if df_1['select'].isna().sum() > 0 or df_2['select'].isna().sum() > 0:
            print(f"Warning: Missing user selections for participant {participant_id}, dropping incomplete rows. Stage 1 missing: {df_1['select'].isna().sum()}, Stage 2 missing: {df_2['select'].isna().sum()}")
            ids1 = df_1[~df_1['select'].isna()]['id']
            ids2 = df_2[~df_2['select'].isna()]['id']
            common_ids = set(ids1).intersection(set(ids2))
            df_1 = df_1[df_1['id'].isin(common_ids)].reset_index(drop=True)
            df_2 = df_2[df_2['id'].isin(common_ids)].reset_index(drop=True)
            print(f"After dropping, remaining records: {len(df_1)}")

        # verify alignment of blockies, true labels, and AI predictions
        assert all(df_1['id'] == df_2['id']), "Mismatched block IDs between stages"
        assert all(df_1['true'] == df_2['true']), "Mismatched true labels between stages"
        assert all(df_1['pred'] == df_2['pred']), "Mismatched AI predictions between stages"

        # extract data for new dataframe
        pdata = dict(
            part_id = [participant_id] * len(df_1),
            blocky_id = df_1['id'].tolist(),
            stakes_level = [stakes_level] * len(df_1),
            y_true = df_1['true'].astype(int).tolist(),
            y_pred = df_1['pred'].astype(int).tolist(),
            y_user_base = df_1['select'].astype(int).tolist(),
            y_user_advised = df_2['select'].astype(int).tolist(),
            time_taken_base = df_1['updated'] - df_1['accessed_updated'],
            time_taken_advised = df_2['updated'] - df_2['accessed_updated']
        )

        return pd.DataFrame(pdata)
    else:
        print(f"Results file for participant {participant_id} not found in {stakes_path}")
        return pd.DataFrame()


def main(basepath, output_path):
    # Placeholder for the main logic to get results

    for stakes_level, stakes_folder in STAKES_FOLDERS.items():
        stakes_path = basepath / stakes_folder
        # get ids from stage 2 to ensure all participants have both stages
        participant_ids = [_get_id(f) for f in (stakes_path / STAGES['2'] / 'results').rglob('results_*.csv') 
                           if _get_id(f) not in FILTERS and _get_id(f) not in REJECTED]
        print(stakes_path)
        print(f"Processing stakes level: {stakes_level}, participants: {len(participant_ids)}")
        all_results = []
        for pid in participant_ids:
            df = process_participant_results(pid, stakes_level, stakes_path)
            if not df.empty:
                all_results.append(df)

        combined_df = pd.concat(all_results, ignore_index=True)
        output_file = output_path / f'per_sample_results_{stakes_level}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined results for stakes level {stakes_level} to {output_file} for {len(combined_df)} records.")

if __name__ == '__main__':
    
    basepath = Path('/Users/djohnson/zScieboArchive/1. Research/human-ai collab/ijcai25/data')

    output_path = Path('output') / 'preprocessed_results'
    output_path.mkdir(parents=True, exist_ok=True)

    main(basepath, output_path)