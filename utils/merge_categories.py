'''
Script to merge a COCO annotation file's categories and update corresponding bounding boxes
'''
# pylint: disable=no-value-for-parameter
# pylint: disable=import-error,no-name-in-module
import os
import json
import click

def get_mapping(specs, category_groups):
    '''Get mapping from the being-merged categories' id to the merge-into one's id'''
    spec_categories = {category['name']: category for category in specs['categories']}
    mapping = {}

    # Check for valid category names
    for category_group in category_groups.split('|'):
        valid_categories = [cat for cat in category_group.split(',') if cat in spec_categories]
        if len(valid_categories) <= 1:
            print('Please enter at least two valid category names')
            continue

        # Map all categories to the 1st one
        for cat in valid_categories:
            mapping[spec_categories[cat]['id']] = spec_categories[valid_categories[0]]['id']

    return mapping

def update_categories(specs, mapping):
    '''Remove being-merged categories'''
    # We loop over specs['categories'] and remove the ones we want to merge into another one
    categories = []
    for category in specs['categories']:
        if category['id'] not in mapping or mapping[category['id']] == category['id']:
            categories.append(category)

    return categories

def update_annotations(specs, mapping):
    '''Update annotations' category_id into the new one for the categories being merged'''
    annotations = []
    for annotation in specs['annotations']:
        if annotation['category_id'] not in mapping or \
            mapping[annotation['category_id']] == annotation['category_id']:
            annotations.append(annotation)
        else:
            annotation['category_id'] = mapping[annotation['category_id']]
            annotations.append(annotation)

    return annotations

@click.command()
@click.option('--input-file', help='COCO annotation json file')
@click.option('--category_groups', help='Category groups to merge \
                    within group with at least two categories per group,\
                    separated by commas. The 1st category is the one to merge into. \
                    Groups are separated by |. \
                    Example: mackerel,horse mackerel,spanish mackerel|squid,calamary\
            ')
@click.option('--output-path', default='ssl_echograms', help='Path to save extracted swarms')
def main(input_file, category_groups, output_path):
    '''main'''
    # read the specs
    with open(input_file, 'r', encoding='utf-8') as file:
        specs = json.load(file)

    # Get mapping from the being-merged categories' id to the merge-into one's id
    mapping = get_mapping(specs, category_groups)
    if len(mapping) == 0:
        print('No merging needed, pls specify at least one group with two or more categories!!!')
        return

    # update the specs' categories
    new_categories = update_categories(specs, mapping)
    specs['categories'] = new_categories

    # Update annotations' category_id into the new one for the categories being merged
    new_annotations = update_annotations(specs, mapping)
    # update the specs
    specs['annotations'] = new_annotations

    # write to file
    with open(os.path.join(output_path,
                           'annotations_merged.json'),
                           'w', encoding='utf-8') as file:
        json.dump(specs, file, indent=4)

if __name__ == '__main__':
    main()
