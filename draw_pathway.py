import matplotlib.pyplot as plt
import numpy as np

def draw_pathway(
    person_locations=None, person_fits=None, person_se=None,
    item_locations=None, item_fits=None, item_se=None, item_labels=None,
    base_size=100, figsize=(12, 10)
):
    """
    Draws a Bond & Fox Pathway Plot (Fit vs. Measure).
    
    Parameters:
    - person_locations: Array of Person measures (theta).
    - person_fits: Array of Person Infit t-stats.
    - person_se: Array of Person Standard Errors.
    - item_locations: Array of Item measures (delta).
    - item_fits: Array of Item Infit t-stats.
    - item_se: Array of Item Standard Errors.
    - item_labels: List of item names.
    - base_size: Helper to scale the marker sizes (defaults to 100).
    - figsize: Tuple for figure size.
    """
    
    plt.figure(figsize=figsize)
    
    # Determine a reference SE for scaling sizes consistently across Persons and Items
    # to ensure 'base_size' means roughly the same thing.
    # We'll use the mean of whatever SEs are provided.
    all_se = []
    if person_se is not None:
        all_se.extend(person_se)
    if item_se is not None:
        all_se.extend(item_se)
    
    if len(all_se) > 0:
        mean_se = np.mean(all_se)
    else:
        mean_se = 1.0 # Default if no SE provided

    # --- Plot Persons ---
    if person_locations is not None and person_fits is not None:
        if person_se is not None:
            # s is Area. Area ~ Radius^2. User wants Radius ~ SE.
            # So s ~ SE^2.
            sizes_p = ((person_se / mean_se) ** 2) * base_size
        else:
            sizes_p = base_size
            
        plt.scatter(person_fits, person_locations, s=sizes_p, c='blue', marker='s', alpha=0.4, label=r'Persons (Size $\propto$ SE)')

    # --- Plot Items ---
    if item_locations is not None and item_fits is not None:
        if item_se is not None:
             sizes_i = ((item_se / mean_se) ** 2) * base_size
        else:
             sizes_i = base_size

        plt.scatter(item_fits, item_locations, s=sizes_i, c='hotpink', marker='o', edgecolors='black', alpha=0.6, label=r'Items (Size $\propto$ SE)')

        # Labels
        if item_labels is not None:
            for i, txt in enumerate(item_labels):
                if i < len(item_locations):
                    plt.text(item_fits[i], item_locations[i], f"  {txt}", fontsize=8, alpha=0.8)

    # Reference Lines
    plt.axvline(-2, color='gray', linestyle='--', linewidth=1)
    plt.axvline(2, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linewidth=0.5)

    # Decoration
    title_str = 'Bond & Fox Pathway Plot (Fit vs. Measure)'
    if len(all_se) > 0:
        title_str += r'\nSymbol Radius $\propto$ Standard Error (Larger = Less Precise)'
        
    plt.title(title_str, fontsize=15)
    plt.ylabel('Logits (Measure) \n(Higher = More Ability / More Difficulty)', fontsize=12)
    plt.xlabel('Infit t-statistic\n(Negative = Overfit/Redundant, Positive = Underfit/Noisy)', fontsize=12)
    plt.xlim(-4, 4) 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()