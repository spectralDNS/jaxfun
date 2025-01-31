import runpy

demos = [
    'poisson1D',
    'poisson1D_curv',
    'poisson2D',
    'poisson2D_curv',
    'poisson3D',
    'poisson1D_periodic',
    'poisson2D_periodic',
    'biharmonic2D'
]

def test_demos():
    for demo in demos:
        try:
            runpy.run_path(f'examples/{demo}.py', run_name='__main__')
        except SystemExit:
            pass

if __name__ == '__main__':
    test_demos()
