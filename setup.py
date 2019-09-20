from setuptools import setup

exec(open('differential_evolution/version.py').read())

setup(
    name='differential_evolution',
    version=__version__,
    description='Differential Evolution Algorithm with OpenMDAO Driver',
    author='Daniel de Vries',
    author_email="danieldevries6@gmail.com",
    packages=['differential_evolution'],
    extras_require={
        'openmdao': ['openmdao>=2.8'],
    },
    install_requires=['numpy>=1.17'],
    tests_require=['openmdao>=2.8', 'pytest', 'pytest-pep8'],
    url='https://github.com/daniel-de-vries/differential-evolution',
    download_url='https://github.com/daniel-de-vries/differential-evolution/archive/v{0}.tar.gz'.format(__version__),
    keywords=['optimization', 'black-box', 'data science', 'differential', 'evolution', 'evolutionary', 'algorithms'],
    license='MIT License',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.7'],
)