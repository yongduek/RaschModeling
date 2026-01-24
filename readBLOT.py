import pandas as pd
import os

def read_blot_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse Header
    metadata = {}
    header_end_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('&END'):
            header_end_idx = i
            break
        if '=' in line:
            # Handle comments after ;
            clean_line = line.split(';')[0]
            if '=' in clean_line:
                parts = clean_line.split('=')
                key = parts[0].strip()
                val = parts[1].strip()
                metadata[key] = val
    
    # Extract metadata with defaults
    try:
        num_items = int(metadata.get('NI', 35))
        item_start_col = int(metadata.get('ITEM1', 5))
        name_start_col = int(metadata.get('NAME1', 1)) 
        name_len = int(metadata.get('NAMELEN', 3))
    except ValueError as e:
        print(f"Error parsing metadata: {e}")
        return None

    # Parse Item Labels
    item_labels = []
    data_start_idx = 0
    found_labels_end = False
    
    for i in range(header_end_idx + 1, len(lines)):
        line = lines[i].strip()
        if line == "END LABELS":
            data_start_idx = i + 1
            found_labels_end = True
            break
        if line:
            # Clean label
            parts = line.split(maxsplit=1)
            if len(parts) > 1 and parts[0].isdigit():
                item_labels.append(parts[1])
            else:
                item_labels.append(line)
    
    # Strict enforcement of item count
    if len(item_labels) > num_items:
        print(f"Warning: Found {len(item_labels)} labels but NI={num_items}. Truncating.")
        item_labels = item_labels[:num_items]
    elif len(item_labels) < num_items:
        print(f"Warning: Found {len(item_labels)} labels but NI={num_items}. Padding.")
        while len(item_labels) < num_items:
            item_labels.append(f"Item_{len(item_labels)+1}")

    # Parse Data
    data = []
    person_ids = []
    
    for i in range(data_start_idx, len(lines)):
        line = lines[i]
        if not line.strip(): continue
        
        # Name
        if len(line) < name_start_col - 1:
            continue
        p_id = line[name_start_col-1 : name_start_col-1+name_len].strip()
        
        # Items
        row_items = []
        curr_col = item_start_col - 1
        for j in range(num_items):
            if curr_col < len(line):
                val = line[curr_col]
                if val.isdigit():
                    row_items.append(int(val))
                else:
                    row_items.append(None) # missing
            else:
                row_items.append(None)
            curr_col += 1
            
        person_ids.append(p_id)
        data.append(row_items)

    df = pd.DataFrame(data, columns=item_labels)
    df.insert(0, 'PersonID', person_ids)
    return df

if __name__ == "__main__":
    df = read_blot_data('blot.txt')
    if df is not None:
        print("DataFrame head:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
