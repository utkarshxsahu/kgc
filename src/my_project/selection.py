import pandas as pd
import random
from my_project.config import T_QUERY, Q4_CAP, MAX_FULL_TRY, NUM_BINS

def compute_entities(df_uris):
    out_deg = df_uris['sub_label'].value_counts()
    in_deg = df_uris['obj_label'].value_counts()
    all_ents = set(df_uris['sub_label']).union(df_uris['obj_label'])
    ent_df = pd.DataFrame({'entity': list(all_ents)})
    ent_df['out_degree'] = ent_df['entity'].map(out_deg).fillna(0).astype(int)
    ent_df['in_degree']  = ent_df['entity'].map(in_deg).fillna(0).astype(int)
    ent_df['total_degree'] = ent_df['in_degree'] + ent_df['out_degree']

    # stratify into Q1..Qn
    cut = pd.qcut(ent_df['total_degree'], q=NUM_BINS, labels=[f"Q{i}" for i in range(1,NUM_BINS+1)], duplicates='drop')
    ent_df['stratum'] = cut
    return ent_df

def select_triplets(df_uris, entities):
    def get_ids(e):
        s = df_uris.loc[df_uris['sub_label']==e, 'triplet_id']
        o = df_uris.loc[df_uris['obj_label']==e, 'triplet_id']
        return set(s).union(o)

    pool = entities.sample(frac=1, random_state=43)['entity']
    selected = set()
    E_final = []
    skipped_full = 0

    for e in pool:
        if len(selected) >= T_QUERY:
            break
        ids = list(get_ids(e))
        if entities.loc[entities['entity']==e, 'stratum'].iloc[0] == 'Q4' and len(ids) > Q4_CAP:
            ids = random.sample(ids, Q4_CAP)

        new_ids = [i for i in ids if i not in selected]
        remain = T_QUERY - len(selected)

        if len(new_ids) <= remain:
            selected.update(new_ids)
            E_final.append(e)
            skipped_full = 0
        else:
            skipped_full += 1
            if skipped_full > MAX_FULL_TRY and remain > 0:
                selected.update(new_ids[:remain])
                E_final.append(e)
                break

    return list(selected), E_final
