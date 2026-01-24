import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def draw_wright_map(person_abilities, item_difficulties, item_names):
    """
    Draws a Wright Map (Variable Map) for Rasch Analysis.
    Rotated to be horizontal (Logits on X-axis).
    
    Parameters:
    - person_abilities: array-like of person ability estimates (logits)
    - item_difficulties: array-like of item difficulty estimates (logits)
    - item_names: list of item names corresponding to item_difficulties
    """
    
    # Create DataFrames for plotting
    persons_df = pd.DataFrame({'Ability': person_abilities})
    
    # Create item labels with Index + Name
    item_labels = [f"{i+1} {name}" for i, name in enumerate(item_names)]
    items_df = pd.DataFrame({'Difficulty': item_difficulties, 'ItemLabel': item_labels})

    # Setup the figure (Horizontal Layout)
    # 2 rows, 1 col. Top: Persons. Bottom: Items. Shared X.
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0})

    # --- Top Panel: Person Ability Distribution ---
    # Histogram x=Ability
    sns.histplot(x=person_abilities, ax=ax_top, color='skyblue', kde=True, bins=20, stat='count')
    ax_top.set_title(f'Person Ability Distribution (N={len(person_abilities)})', fontsize=12)
    ax_top.set_ylabel('Count')
    ax_top.grid(axis='x', linestyle='--', alpha=0.3)

    # Add mean person ability line
    mean_ability = np.mean(person_abilities)
    ax_top.axvline(mean_ability, color='blue', linestyle='--', label=f'Mean Person ({mean_ability:.2f})')
    ax_top.legend(loc='upper right')

    # --- Bottom Panel: Item Difficulty Hierarchy ---
    ax_bottom.set_title(f'Item Difficulty Pathway (N={len(item_difficulties)})', fontsize=12)
    ax_bottom.set_xlabel('Logits (Measure)')
    ax_bottom.set_ylabel('')
    ax_bottom.set_yticks([])
    
    ax_bottom.spines['left'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)

    # Plot each item on X axis
    # To place markers between histogram axis and logits axis:
    # We set limits such that y=0 (markers) is at the TOP of the bottom plot.
    sns.stripplot(x='Difficulty', data=items_df, ax=ax_bottom, color='red', size=8, jitter=False)
    
    # Adjust Y-limits to push markers to the top
    # Default is approx -0.5 to 0.5. We change to e.g. -1.0 to 0.1 so 0 is near the top edge.
    ax_bottom.set_ylim(-1, 0.1)

    # Annotate items
    # Text below the dots, rotated 90 degrees (reading down)
    for _, row in items_df.iterrows():
        # y=-0.05 puts it below the dot.
        # va='top' means the top of the text is anchored at y.
        # rotation=90 usually reads bottom-to-top. -90 or 270 reads top-to-bottom.
        # Let's use 270 (top-to-bottom) so it flows down away from the axis?
        # Or 90 (bottom-to-top) is more standard for vertical axis labels.
        # Let's stick to 90 (standard vertical text).
        ax_bottom.text(row['Difficulty'], -0.05, f" {row['ItemLabel']}", 
                    rotation=90, va='top', ha='center', fontsize=9, color='darkred')
    
    # Add mean item difficulty line
    mean_item = np.mean(item_difficulties)
    ax_bottom.axvline(mean_item, color='red', linestyle=':', label=f'Mean Item ({mean_item:.2f})')
    ax_bottom.legend(loc='lower right')
    
    # Global Title
    plt.suptitle('Wright Map: The Construct Pathway (Rotated)', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.show()