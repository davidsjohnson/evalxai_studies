from pathlib import Path

import pandas as pd

FILTERS = ['study1-59', 'study1-60']
REJECTED = ['study1-53', 'study1-54', 'study1-55', 'study1-56', 'study1-57', 'study1-58', 'study1-59']

# new dataframe structure one dataframe per stage: part_id, blocky_id, stakes_level, true, ai_pred, user_pred, time_taken

def _get_id(res_file: Path):
    return res_file.stem.split('_')[-1]

def proces_participant_results(participant_id, stage_path, stage_name, stakes_level):
    """ extracts results from participant results file into dataframe """
    results_file = stage_path / f'results_{stage_name}_{participant_id}.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        pdata = dict(
            part_id = [participant_id] * len(df),
            blocky_id = df['id'].tolist(),
            stakes_level = [stakes_level] * len(df),
            true = df['true'].astype(int).tolist(),
            ai_pred = df['pred'].astype(int).tolist(),
            user_pred = df['select'].astype(int).tolist(),
            time_taken = df['updated'] - df['accessed_updated']
        )

        return pd.DataFrame(pdata)
    else:
        print(f"Results file for participant {participant_id} not found in {stage_path}")
        return pd.DataFrame()

def main(basepath, output_path, stages, stakes):
    # Placeholder for the main logic to get results

   
    for stage, stage_name in stages.items():
        stage_results = []
        for stake_level, stake_folder in stakes.items():
        
            stage_path = basepath / stake_folder / stage_name / 'results'
            participant_ids = [_get_id(f) for f in stage_path.rglob('results_*.csv') 
                               if _get_id(f) not in FILTERS and _get_id(f) not in REJECTED]
            print(stage_path)
            print(f"Processing stage: {stage_name}, stake level: {stake_level}, participants: {len(participant_ids)}")
            for pid in participant_ids:
                df = proces_participant_results(pid, stage_path, stage_name, stake_level)
                if not df.empty:
                    stage_results.append(df)

        stage_df = pd.concat(stage_results, ignore_index=True)
        output_file = output_path / f'combined_results_{stage_name}.csv'
        stage_df.to_csv(output_file, index=False)
        print(f"Saved combined results for stage {stage_name} to {output_file} for {len(stage_df)} records.")
        

        

if __name__ == '__main__':
    
    basepath = Path('/Users/djohnson/zScieboArchive/1. Research/human-ai collab/ijcai25/data')
    stages = {'0': 'training', '1': 'validation_nodiag', '2': 'validation_noxai'}
    stakes = {'high': 'example_validation_study',
              'low': 'example_validation_study_ls'}
    
    output_path = Path('output') / 'preprocessed_results'
    output_path.mkdir(parents=True, exist_ok=True)

    main(basepath, output_path, stages, stakes)