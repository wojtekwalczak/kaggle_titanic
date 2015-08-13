import sys

def pick_pipeline(argv):
    if argv[1] == 'gbc':
        from titanic.pipelines.gbc_pipeline import pipeline
    elif argv[1] == 'lr':
        from titanic.pipelines.lr_pipeline import pipeline
    else:
        print('USAGE: {} [gbc | lr]'.format(sys.argv[0]))
        sys.exit(1)
    return pipeline