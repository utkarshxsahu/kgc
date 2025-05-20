import pandas as pd

def load_templates(template_path):
    # expects two columns: rel_label, prompt_template
    return pd.read_csv(template_path).set_index('rel_label')['prompt_template'].to_dict()

def build_prompts(selected_ids, df_kg, templates):
    sub = df_kg.set_index('triplet_id')['sub_label']
    obj = df_kg.set_index('triplet_id')['obj_label']
    rel = df_kg.set_index('triplet_id')['rel_label']

    rows = []
    for tid in selected_ids:
        t = templates.get(rel[tid])
        if not t:
            raise KeyError(f"No template for relation {rel[tid]}")
        prompt = t.replace('{subject}', sub[tid]).replace('{object}', obj[tid])
        rows.append({'triplet_id': tid, 'triplet_prompt': prompt})
    return pd.DataFrame(rows)
