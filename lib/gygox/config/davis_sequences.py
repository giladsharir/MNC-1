# davis will hold a list of all the sequences in the DAVIS dataset, as well as
# their test and train subsets
davis = {}
# These are all the sequences in DAVIS
davis['all_sequences'] = [
    'bear', 'blackswan', 'bmx-bumps', 'bmx-trees', 'boat',
    'breakdance', 'breakdance-flare', 'bus', 'camel',
    'car-roundabout', 'car-shadow', 'car-turn', 'cows',
    'dance-jump', 'dance-twirl', 'dog', 'dog-agility',
    'drift-chicane', 'drift-straight', 'drift-turn', 'elephant',
    'flamingo', 'goat', 'hike', 'hockey', 'horsejump-high',
    'horsejump-low', 'kite-surf', 'kite-walk', 'libby', 'lucia',
    'mallard-fly', 'mallard-water', 'motocross-bumps',
    'motocross-jump', 'motorbike', 'paragliding',
    'paragliding-launch', 'parkour', 'rhino', 'rollerblade',
    'scooter-black', 'scooter-gray', 'soapbox', 'soccerball',
    'stroller', 'surf', 'swing', 'tennis', 'train']
# These are the validation sequences in DAVIS (what we generally run)
davis['val_sequences'] = [
    'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog',
    'drift-chicane', 'drift-straight', 'goat', 'horsejump-high',
    'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch',
    'parkour', 'scooter-black', 'soapbox']
davis['train_sequences'] = list(
    set(davis['all_sequences']).difference(set(davis['val_sequences'])))
