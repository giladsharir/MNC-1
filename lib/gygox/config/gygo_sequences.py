# davis will hold a list of all the sequences in the DAVIS dataset, as well as
# their test and train subsets
gygo = {}
# These are all the sequences in GyGO
gygo['all_sequences'] = []
# These are the validation sequences in GyGO (what we generally run)
gygo['val_sequences'] = [
    'seg_prod_t01/0x1TFTRZ5NGJ',
    'seg_prod_t01/0xVY2GGZQ2KS',
    'seg_prod_t01/0xYZHA3B17KL',
    'seg_prod_t02/0x4I1RJ8X8KI',
    'seg_prod_t02/0x81A4H01HZ7',
    'seg_prod_t02/0xWATCHCOOL',
    'seg_prod_t02/OxCOOLCAR',
    'seg_prod_t03/0x0FOHIX8D37',
    'seg_prod_t03/0x5GNUU3N5NH',
    'seg_prod_t03/0x7ZZJZFW460',
    'seg_prod_t03/0xLL2MDBWK8J',
    'seg_prod_t03/0xNCEAPSH35F',
    'seg_prod_t03/0xRBRRVRLMG4',
    'seg_prod_t03/0xWIQ8DJ0YSG',
    'seg_prod_t04/1x002651',
    'seg_prod_t04/1x003436',
    'seg_prod_t04/1x003717',
    'seg_prod_t04/1x004055',
    'seg_prod_t04/1x004657',
    'seg_prod_t04/1x005044']
gygo['train_sequences'] = list(
    set(gygo['all_sequences']).difference(set(gygo['val_sequences'])))
